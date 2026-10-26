import torch
from datetime import datetime
import os
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import copy

from sklearn.model_selection import KFold
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, auc, brier_score_loss
from sklearn.preprocessing import label_binarize


import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, max_epochs=3, optim=None, loss_fn=nn.CrossEntropyLoss(), lr=0.03, k_folds=4, gpu=None, workers=0, batch_size=256):
        self.max_epochs = max_epochs
        self.gpu = gpu
        self.workers = workers
        self.batch_size = batch_size

        self.optim = optim
        self.loss_func = loss_fn
        self.lr = lr

        self.k_folds = k_folds

        # storing loss per train step
        self.train_loss_store = []
        self.val_loss_store = []

        # storing loss per fold
        self.kfold_train = []
        self.kfold_val = []

        # label mapping
        self.label_to_idx_map = {}

        #saving path
        current_datetime = datetime.now().strftime('%d-%m-%y_%H-%M')
        self.save_path = os.path.join("models", current_datetime)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def prepare_model(self, model):
        self.model = model
        if self.gpu:
            self.model.to(self.gpu)


    def load_model(self, model_object, path):
        model_object.load_state_dict(torch.load(path, weights_only=True))
        self.prepare_model(model_object)


    def loss_plots(self):
        avg_train_loss, avg_val_loss = self.get_average_loss()

        plt.plot(avg_train_loss, label="Train Loss")
        plt.plot(avg_val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train-Validation Loss")
        plt.legend()

        plot_path = os.path.join(self.save_path, "train-val-loss.png")
        plt.savefig(plot_path, bbox_inches='tight')
        

    def init_model_weights(self, train_dataloader):

        # checking for any uninitialised parameters
        needs_init = any(
            nn.parameter.is_lazy(p) for p in list(self.model.parameters()) + list(self.model.buffers())
        )

        if needs_init:
            # ===============================
            # need to use dummy batch to initialise
            # lazyLinear layer
            for batch, _ in train_dataloader:
                if self.gpu:
                    batch = batch.to(self.gpu)
                dummy_input = batch[:2]
                break

            self.model.eval()
            with torch.no_grad():
                _ = self.model(dummy_input)
            # ===============================

        def init_weights(m):
            if isinstance(m, (nn.LazyLinear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.model.apply(init_weights)


    def init_optimiser(self):
        if isinstance(self.optim, (torch.optim.Adam)) or self.optim is None:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            print("No optimiser found.")


    def get_average_loss(self):
        # get average (median) loss for train and val
        train_loss_mean = np.median(self.kfold_train, axis=0)
        val_loss_mean = np.median(self.kfold_val, axis=0)
        return train_loss_mean, val_loss_mean


    def fit(self, train_dataset, model, verbose=False):
        self.label_to_idx_map = train_dataset.label_to_idx
        self.prepare_model(model)
        self.init_model_weights(DataLoader(train_dataset, batch_size=16))
        initial_model_state = copy.deepcopy(self.model.state_dict())
        
        
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):

            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)

            train_dataloader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=bool(self.gpu))
            val_dataloader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=bool(self.gpu))

            # reset model params
            self.model.load_state_dict(initial_model_state)
            self.init_optimiser()
            
            for epoch in range(1,self.max_epochs+1):
                train_loss_accumulator = 0
                val_loss_accumulator = 0

                self.model.train()
                for x_train_batch, y_train_batch in train_dataloader:

                    if self.gpu:
                        x_train_batch = x_train_batch.to(self.gpu)
                        y_train_batch = y_train_batch.to(self.gpu)

                    self.optim.zero_grad()
                    model_prediction = self.model(x_train_batch)
                    loss_train = self.loss_func(model_prediction, y_train_batch)
                    train_loss_accumulator += loss_train.item()
                    loss_train.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optim.step()

                epoch_avg_train_loss = train_loss_accumulator/len(train_dataloader)
                self.train_loss_store.append(epoch_avg_train_loss)

                self.model.eval()
                for x_val_batch, y_val_batch in val_dataloader:

                    if self.gpu:
                        x_val_batch = x_val_batch.to(self.gpu)
                        y_val_batch = y_val_batch.to(self.gpu)

                    with torch.no_grad():
                        model_prediction = self.model(x_val_batch)
                        loss_val = self.loss_func(model_prediction, y_val_batch)
                        val_loss_accumulator += loss_val.item()

                epoch_avg_val_loss = val_loss_accumulator/len(val_dataloader)
                self.val_loss_store.append(epoch_avg_val_loss)

                if verbose:
                    print(f"Epoch {epoch}\nTrain_Loss={epoch_avg_train_loss}\nVal_loss={epoch_avg_val_loss}")

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.kfold_train.append(self.train_loss_store)
            self.kfold_val.append(self.val_loss_store)
            self.train_loss_store = []
            self.val_loss_store = []

            # save model
            model_save_path = os.path.join(self.save_path, f"Model{fold+1}.pt")
            self.model.save(model_save_path)

        self.kfold_train = np.array(self.kfold_train)
        self.kfold_val = np.array(self.kfold_val)

        # save loss plots
        self.loss_plots()


    def get_list_of_models(self, custom_path):
        models = []
        if os.path.isfile(custom_path):
            models.append(custom_path)
        else:
            files = os.listdir(custom_path)
            for model_filename in files:
                if model_filename.endswith(".pt"):
                    models.append(os.path.join(custom_path, model_filename))
        return models


    def predict_single_image(self, img, model_object, custom_path=None):
            if custom_path is None:
                custom_path = self.save_path
    
            model_paths = self.get_list_of_models(custom_path)
            preds = []
            probs = []
    
            for model_path in model_paths:
                self.load_model(model_object, model_path)
                self.model.eval()
    
                with torch.no_grad():
                    logits = self.model(img)
                    prob = torch.softmax(logits, dim=1)
                    class_pred = torch.argmax(logits, dim=1)
    
                    preds.append(class_pred)
                    probs.append(prob)
    
            return torch.stack(preds), torch.stack(probs)

    
    def test_model(self, test_dataset, model_object, custom_path=None):

        if custom_path is None:
            custom_path = self.save_path

        model_paths = self.get_list_of_models(custom_path)

        all_y_preds = []
        all_y_probs = []

        for model_path in model_paths:
            self.load_model(model_object, model_path)
            y_preds = []
            y_true = []
            y_preds_probs = []
            self.model.eval()
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers)

            for x_batch, y_batch in test_dataloader:
                if self.gpu:
                    x_batch = x_batch.to(self.gpu)

                with torch.no_grad():
                    logits = self.model(x_batch)
                    probs = torch.softmax(logits, dim=1)
                    class_pred = torch.argmax(logits, dim=1)


                y_preds.append(class_pred.cpu().numpy())
                y_true.append(y_batch.numpy())
                y_preds_probs.append(probs.cpu().numpy())
                
            y_preds_flat = np.concatenate(y_preds)
            all_y_preds.append(y_preds_flat)

            y_preds_probs_flat = np.concatenate(y_preds_probs)
            all_y_probs.append(y_preds_probs_flat)

        all_y_preds = np.array(all_y_preds)
        all_y_probs = np.array(all_y_probs)
        y_true = np.concatenate(y_true)

        return self.evaluate(all_y_preds, all_y_probs, y_true)


    def evaluate(self, y_pred, y_probs, y_true):

        # get most common predictions
        mode_predictions = stats.mode(y_pred, axis=0).mode
        mean_probs = np.mean(y_probs, axis=0)
        
        #acc, prec, rec, f1, classification report, confusion matrix
        acc = accuracy_score(y_true, mode_predictions)
        precision = precision_score(y_true, mode_predictions, average='macro')
        recall = recall_score(y_true, mode_predictions, average='macro')
        f1 = f1_score(y_true, mode_predictions, average='macro')
        report = classification_report(y_true, mode_predictions)
        conf_matrix = confusion_matrix(y_true, mode_predictions, normalize='pred')

        fig, ax = plt.subplots(figsize=(10,10))

        matrix_display = ConfusionMatrixDisplay(conf_matrix, display_labels=list(self.label_to_idx_map))
        matrix_display.plot(ax=ax)
        matrix_display.ax_.tick_params(axis='x', labelrotation=70)
        plt.tight_layout()
        conf_matrix_path = os.path.join(self.save_path, "Confusion_matrix.png")
        plt.savefig(conf_matrix_path)

        roc_plot = self.create_roc_plot(y_true, mean_probs, self.model.num_classes)
        
        brier_score = brier_score_loss(y_true, mean_probs)

        return acc, precision, recall, f1, report, conf_matrix, roc_plot, brier_score


    def create_roc_plot(self, y_true, y_probs, num_classes):
            overall_auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
            classes = list(range(num_classes))
            y_true_bin = label_binarize(y_true, classes=classes)
    
            # Create Figure and Axes objects
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                class_auc = auc(fpr, tpr)
                # Use ax.plot instead of plt.plot
                ax.plot(fpr, tpr, lw=2, label=f"Class {classes[i]} AUC={class_auc:.2f}")
    
            ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random Guess')
            
            # Use ax.set_* methods for formatting
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'Multiclass ROC Curves (Overall AUC: {overall_auc:.4f})')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
            
            # Close the plot to prevent it from displaying immediately in interactive environments (like Jupyter)
            roc_plot_path = os.path.join(self.save_path, "roc_auc.png")
            plt.savefig(roc_plot_path)

            return fig