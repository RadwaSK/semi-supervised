from models.ViT import ViT
from utils import get_test_dataloader
from os.path import join
import os
import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score

os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--data_path', type=str, default='dataset', help='path for Labels csv files')
parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch size for data')
parser.add_argument('-mn', '--model_name', type=str, required=True, help='enter name of model to test from '
                                                                         'models in saved_models')

opt = parser.parse_args()

test_path = join(opt.data_path, 'test')

dataloader = get_test_dataloader(opt.batch_size, test_path)

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
print('CUDA available:', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

data_size = len(dataloader.dataset)
print("Number of data:", data_size)

model_path = 'saved_models'
model = ViT(use_cuda).to(device)
model_name = join(model_path, opt.model_name)
checkpoint = torch.load(model_name)
model.load_state_dict(checkpoint['model_state'])
model.eval()

y_trues = np.empty([0])
y_preds = np.empty([0])

model.eval()

for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.long().squeeze().to(device)

    outputs = model(inputs).squeeze()

    preds = torch.max(outputs, dim=-1)
    print(preds)
    y_trues = np.append(y_trues, labels.data.cpu().numpy())
    y_preds = np.append(y_preds, preds.indices.cpu())


print('\nF1 Score: \n' + str(f1_score(y_trues, y_preds, average='weighted')))
print('\nRecall Score: \n' + str(recall_score(y_trues, y_preds, average='weighted')))
print('\nPrecision Score: \n' + str(precision_score(y_trues, y_preds, average='weighted')))
print('\nTesting Accuracy: \t' + str(accuracy_score(y_trues, y_preds)))
print('\nConfusion Matrix: \n' + str(confusion_matrix(y_trues, y_preds)))

