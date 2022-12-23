import torchvision.transforms as transforms
from torch.utils.data import Dataset
from random import sample
from os.path import join
import os
import cv2


class InferenceDataset(Dataset):
    def __init__(self, data_path, ratio=None, transform=None):
        self.root = data_path
        paths = os.listdir(data_path)

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        if ratio:
            count = ratio * len(paths)
            self.paths = sample(paths, count)
        else:
            self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = join(self.root, self.paths[index])
        img = cv2.imread(path)
        data = self.transform(img)
        return data, path
