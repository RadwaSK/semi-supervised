from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision import datasets
import torch
from models.ViT import ViT
from random import sample
import numpy as np
import pandas as pd
import os
from utils import get_inference_dataloader_v2


# os.makedirs('saved_models', exist_ok=True)

def predict_feature_vec(dataloader, model_name, labeled = True):
    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    print('CUDA available:', use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    model_path = 'saved_models'
    model_name_ = os.path.join(model_path, model_name)
    classes_num = 810  # len(classes)

    model = ViT(use_cuda, classes_num).to(device)
    checkpoint = torch.load(model_name_)
    model.load_state_dict(checkpoint['model_state'])

    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    y_trues = np.empty([0])
    features_vec = np.empty([1000])
    model.eval()

    if labeled:
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).squeeze()
            #     print(outputs.shape)
            features_vec = np.append(features_vec, outputs.data.cpu().numpy())
            y_trues = np.append(y_trues, labels.data.cpu().numpy())

        len_data = int(len(dataloader.dataset))
        features_vec = np.resize(features_vec, (len_data, 1000))
        print(features_vec.shape)

        return features_vec, y_trues
    else:
        unlab_imgs = np.empty([0])

        for inputs, labels, unlab_img in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).squeeze()

            features_vec = np.append(features_vec, outputs.data.cpu().numpy())
            y_trues = np.append(y_trues, labels.data.cpu().numpy())
            unlab_imgs = np.append(unlab_imgs, unlab_img)

        len_data = int(len(dataloader.dataset))
        features_vec = np.resize(features_vec, (len_data, 1000))
        print(features_vec.shape)

        return features_vec, y_trues, unlab_imgs


def generate_feature_vec_train(batch_size, train_path, model_name):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.ImageFolder(train_path, transform = train_transform)
    train_lb_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, drop_last=False)

    lb_features_v, lb_labels = predict_feature_vec(train_lb_loader, model_name)

    if not os.path.isdir("Saved_Features_Vec"):
        os.makedirs("Saved_Features_Vec")
    np.savez_compressed('Saved_Features_Vec/lb_features_v.npz', lb_features_v)
    np.savez_compressed('Saved_Features_Vec/lb_labels.npz', lb_labels)

    return lb_features_v, lb_labels


def generate_feature_vec_utrain(batch_size, utrain_path, model_name, pl_utrain_path):
    ps_ulb = pd.read_csv(pl_utrain_path)

    unlabeled_dataloader = get_inference_dataloader_v2(batch_size, utrain_path, ps_ulb['img_name'].values,
                                                       ps_ulb['folder_id'].values)

    ulb_features_v, ulb_ps, ulabeled_imgs = predict_feature_vec(unlabeled_dataloader, model_name)
    np.savez_compressed('Saved_Features_Vec/ulb_features_v.npz', ulb_features_v)
    np.savez_compressed('Saved_Features_Vec/lb_ps.npz', ulb_ps)

    return ulb_features_v, ulb_ps, ulabeled_imgs

