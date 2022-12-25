from models.ViT import ViT
from utils import get_inference_dataloader
from os.path import join
import os
import pickle
import argparse
import torch
import shutil
import torch.nn.functional as F

os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--data_path', type=str, default='dataset', help='path for Data files')
parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch size for data')
parser.add_argument('-m', '--model_name', type=str, required=True, help='enter name of model to test from '
                                                                        'models in saved_models')
parser.add_argument('-if', '--input_folder', type=str, default='u_train', help='folder name that includes input data')
parser.add_argument('-pf', '--preds_folder', type=str, default='pl_train', help='folder to save predicted data')

opt = parser.parse_args()

test_path = join(opt.data_path, opt.input_folder)

with open('class_to_indx.pkl', 'rb') as f:
    class_to_indx = pickle.load(f)
indx_to_class = {class_to_indx[k]: k for k in class_to_indx.keys()}

dataloader = get_inference_dataloader(opt.batch_size, test_path)

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
print('CUDA available:', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

data_size = len(dataloader.dataset)
print("Number of data:", data_size)

model_path = 'saved_models'
model = ViT(use_cuda, 810).to(device)
model_name = join(model_path, opt.model_name)
checkpoint = torch.load(model_name)
model.load_state_dict(checkpoint['model_state'])
model.eval()

model.eval()
predicted_folder = join(opt.data_path, opt.preds_folder)
os.makedirs(predicted_folder, exist_ok=True)

for inputs, paths in dataloader:
    inputs = inputs.to(device)

    outputs = model(inputs).squeeze()

    preds, pred_labels = F.softmax(outputs, 1).max(1)

    preds = preds.data.cpu().numpy()
    pred_labels = pred_labels.data.cpu().numpy()
    for i in range(opt.batch_size):
        if preds.data[i] > 0.7:
            folder = join(predicted_folder, str(indx_to_class[pred_labels[i]]))
            os.makedirs(folder, exist_ok=True)
            print('Moving image: ', paths[i][paths[i].rfind('/'):])
            shutil.move(paths[i], folder)

