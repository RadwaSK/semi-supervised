import torchvision.transforms as transforms
from torch.utils.data import Dataset
from random import sample
from os.path import join
from utils import intersection
import os
import cv2


# class for unlabeled data 
class utrainDataset(Dataset):
    def __init__(self, data_path, ratio=None, transform=None, unlabeled_set = None, ps_labels = None):
        self.root = data_path
        All_paths = os.listdir(data_path)
        paths = intersection(All_paths, unlabeled_set)
        #set(All_paths).intersection(unlabeled_set)
        self.unlabeled_imgs = unlabeled_set
        self.unlabeled_ps = ps_labels
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
        label = self.unlabeled_ps[index]
        unlabeled_img = self.unlabeled_imgs[index]
        return data, label, unlabeled_img