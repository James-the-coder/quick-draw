import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2

from quickdraw import QuickDrawData, QuickDrawDataGroup


class ImageDataset(Dataset):
    def __init__(self, drawings=None, transform=None, label_to_idx=None):
        self.drawings = drawings
        self.transform = transform
        self.label_to_idx = label_to_idx or {}
        self.X, self.y = self.preprocess()

    def preprocess(self):
        X, y = [], []
        for drawing in self.drawings:
            image = drawing.image
            label_str = drawing.name

            if self.transform:
                image = self.transform(image)

            if self.label_to_idx and label_str in self.label_to_idx:
                label_idx = self.label_to_idx[label_str]
            else:
                label_idx = 0

            X.append(image)
            y.append(label_idx)
        return torch.stack(X), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.drawings)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def get_training_transform(augment_p=0.5):
    return transforms.Compose(
        [   
            transforms.Grayscale(1),
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            v2.RandomApply(
            [
                v2.RandomResizedCrop(size=(28,28), antialias=True, scale=(0.5, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
            ],
            p=augment_p
            )
        ]
    )

def get_training_transform_v2(augment_p=0.5):
    return v2.Compose(
        [
            v2.Grayscale(1),
            v2.Resize((28,28), antialias=True),
            transforms.ToTensor(),
            v2.Lambda(lambda x: 1 - x),
            v2.RandomApply(
            [
                #v2.RandomResizedCrop(size=(28,28), antialias=True, scale=(0.5, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomErasing(p=0.5)
            ],
            p=augment_p
            )
        ]
    )

def get_test_transform():
    return transforms.Compose(
        [
            transforms.Resize((28,28)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ]
    )


def get_test_transform_v2():
    return v2.Compose(
        [
            v2.Grayscale(1),
            v2.Resize((28,28), antialias=True),
            transforms.ToTensor(),
            v2.Lambda(lambda x: 1 - x),
        ]
    )

def download_data(classes=8, recognized=True, num_drawings=1000):
    master_drawings = []
    label_to_idx = {}
    qd = QuickDrawData(recognized=recognized)
    class_names = qd.drawing_names[:classes]
    for idx, class_name in enumerate(class_names):
        label_to_idx[class_name] = idx
        group = QuickDrawDataGroup(name=class_name, recognized=recognized, max_drawings=num_drawings, print_messages=False)

        for drawing in group.drawings:
            master_drawings.append(drawing)

    return master_drawings, label_to_idx