from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import time
from PIL import Image
import os


class TrainDataset(Dataset):
    def __init__(self, data_dir):

        dataset_path = os.path.join(data_dir, "digiface1m")
        self.dataset_path = dataset_path
        image_paths = []
        labels = []
        for label_name in os.listdir(dataset_path):
            try:
                for image_path in os.listdir(os.path.join(dataset_path, label_name) ):
                    image_paths.append(os.path.join(dataset_path, label_name, image_path))
                    labels.append(int(label_name))
            except NotADirectoryError:
                pass

        labels = np.array(labels)
        image_paths = np.array(image_paths)
        idx = labels <= 1600 # TODO what split are we going to choose
        self.labels = labels[idx]
        self.images = image_paths[idx]

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(227, scale=(0.08, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.targets = labels
        unique = {}
        for class_idx in np.unique(self.targets):
            # find the idx for each class
            # {class_idx: pos_idx}
            unique[str(class_idx)] = np.where(self.targets == class_idx)[0]

        self.pidxs = []
        self.qidx = []
        for idx, class_idx in enumerate(self.targets):
            # for each position, list all the targets with the same class exept ifself
            pos_idx = unique[str(class_idx)]
            pos_idx = pos_idx[pos_idx != idx]

            # at least one of the same class
            if len(pos_idx) > 0:
                self.pidxs.append(pos_idx)
                self.qidx.append(idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.transform(
            Image.open(self.images[idx]).convert("RGB")
        )
        label = self.labels[idx]


        qidx = self.qidx[idx]
        pidxs = self.pidxs[idx]

        pidx = np.random.choice(pidxs, 1)[0]

        qimg = Image.open(self.images[qidx]).convert("RGB")
        pimg = Image.open(self.images[pidx]).convert("RGB")

        output = [qimg, pimg]
        output = [self.transform(img) for img in output]

        assert self.targets[qidx] == self.targets[pidx]
        target = [self.targets[qidx], self.targets[pidx]]
        

        return output, target


class TestDataset(Dataset):
    def __init__(self, data_dir):

        dataset_path = os.path.join(data_dir, "digiface1m")
        self.dataset_path = dataset_path
        image_paths = []
        labels = []
        for label_name in os.listdir(dataset_path):
            try:
                for image_path in os.listdir(os.path.join(dataset_path, label_name) ):
                    image_paths.append(os.path.join(dataset_path, label_name, image_path))
                    labels.append(int(label_name))
            except NotADirectoryError:
                pass

        labels = np.array(labels)
        image_paths = np.array(image_paths)
        idx = labels > 1600 # TODO what split are we going to choose
        self.labels = labels[idx]
        self.images = image_paths[idx]

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(227, scale=(0.08, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.transform(
            Image.open(self.images[idx]).convert("RGB")
        )
        label = self.labels[idx]

        return image, label
