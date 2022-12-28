from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
from InferenceDataset import InferenceDataset
from unlabeledClass import utrainDataset
import pickle


def get_t_v_dataloaders(batch_size, train_data_root, val_data_root):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.ImageFolder(train_data_root, transform=transform)
    val_dataset = datasets.ImageFolder(val_data_root, transform=val_transform)
    with open('class_to_indx.pkl', 'wb') as f:
        pickle.dump(train_dataset.class_to_idx, f)

    return {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True),
            'validation': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True),
            'classes': train_dataset.classes}


def get_inference_dataloader(batch_size, test_data_root):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = InferenceDataset(test_data_root, transform=transform)

    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


def get_test_dataloader(batch_size, test_data_root):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.ImageFolder(test_data_root, transform=transform)

    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True), \
        test_dataset.classes


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def get_inference_dataloader_v2(batch_size, test_data_root, ps_unlabeled_set, ps_labels):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = utrainDataset(test_data_root, transform=transform, unlabeled_set = ps_unlabeled_set, ps_labels = ps_labels)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

## EarlyStopping class used for stopping the model early while training
class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.1):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0