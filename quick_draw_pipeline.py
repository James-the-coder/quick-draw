from cnn_model import ConvNet, ConvNetV2, ConvNetV3
from image_dataset import ImageDataset, get_training_transform, get_test_transform, download_data
from image_dataset import get_training_transform_v2, get_test_transform_v2
from trainer import Trainer

from sklearn.model_selection import train_test_split
import torch

import time

# ======= sorting out devices =======
def cpu():
    return torch.device('cpu')


def gpu(i=0):
    return torch.device(f'cuda:{i}')


def num_gpus():
    return torch.cuda.device_count()


def try_gpu(i=0):
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()
# ===================================

CLASSES = 5
TEST_SIZE = 0.1
AUGMENT_PROB = 0.05
MY_DEVICE = try_gpu()

if __name__ == '__main__':
    print(f"Using: {MY_DEVICE}")
    print("Downloading Drawings...")
    # ====== download dataset ======
    all_drawings, label_mapping = download_data(classes=CLASSES, num_drawings=8500)
    # ==============================

    print("Preprocessing Data...")
    # ====== preprocess data ======
    train_drawings, test_drawings = train_test_split(all_drawings, test_size=TEST_SIZE, random_state=42, shuffle=True)
    
    train_image_transform = get_training_transform_v2(augment_p=AUGMENT_PROB)
    test_image_transform = get_test_transform_v2()

    image_train_dataset = ImageDataset(drawings=train_drawings, transform=train_image_transform, label_to_idx=label_mapping)
    image_test_dataset = ImageDataset(drawings=test_drawings, transform=test_image_transform, label_to_idx=label_mapping)
    # =============================

    print("Training Model...")
    model = ConvNetV2(classes=CLASSES)
    trainer = Trainer(max_epochs=30, k_folds=5, gpu=MY_DEVICE, batch_size=512, lr=0.01)

    t0 = time.time()
    trainer.fit(train_dataset=image_train_dataset, model=model)
    t1 = time.time()

    print(f"Completed Training: {t1-t0:.2f}s")

    _, _, _, _, report, _, _, brier_score = trainer.test_model(image_test_dataset, model)

    print(report)

    print(f"Brier Score: {brier_score:.3f}")