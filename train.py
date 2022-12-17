from models.ViT import ViT
from utils import get_t_v_dataloaders, EarlyStopping
from os.path import join
import os
import argparse
import torch
from torch import optim
import numpy as np
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import torch.nn.functional as F


os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--data_path', type=str, default='dataset', help='path for Labels csv files')
parser.add_argument('-tt', '--train_type', type=str, default='l', help='Train on labeled data ("l") or not')
parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch size for data')
parser.add_argument('-st', '--st_epoch', type=int, default=0, help='start epoch number')
parser.add_argument('-n', '--n_epochs', type=int, default=200, help='number of epochs')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='learning rate for optimizer')
parser.add_argument('-l', '--load_saved_model', type=str, default='n', help='enter y to load saved model, n for not')
parser.add_argument('-m', '--model_name', type=str, default='saved_models', help='enter name of saved model in '
                                                                                 'saved_models folder')
parser.add_argument('-rn', '--run_name', type=str, required=True, help='Enter run name for folders and models')
parser.add_argument('-r', '--run_num', type=int, default=len(os.listdir('saved_models')), help='Trial Run Number')
parser.add_argument('-o', '--optim', type=str, default='SGD', help='Which Optim to use, "Adam" or "SGD')

opt = vars(parser.parse_args())

print('\n======================================================================\n')
print(opt)
print('\n======================================================================\n')

if opt['train_type'] == 'l':
    train_path = join(opt['data_path'], 'l_train')
elif opt['tt'] == 'u':
    # I don't know what this means now, bs 3mla l condition
    train_path = join(opt['data_path'], 'u_train')
else:
    raise 'Please enter a valid option for  training type'

val_path = join(opt['data_path'], 'val')

datasets = get_t_v_dataloaders(opt['batch_size'], train_path, val_path)
classes = datasets['classes']
n_classes = len(classes)
dataset_sizes = {x: len(datasets[x].dataset) for x in ['train', 'validation']}

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
print('Training on CUDA:', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

model_path = 'saved_models'
freeze = False
pretrained = True

model = ViT(use_cuda, n_classes, pretrained, freeze).to(device)
if opt['optim'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=opt['learning_rate'])
elif opt['optim'] == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt['learning_rate'])
else:
    raise 'Enter a valid optimizer name'

if opt['load_saved_model'] == 'y':
    model_name = join(model_path, opt['model_name'])
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    print('\n<<Done loading model & optimizer>>\n')

experiment_name = opt['run_name'] + '_' + str(opt['run_num'])
new_model_name = join(model_path,  experiment_name + '.pt')
plots_folder = join('plots', experiment_name)
os.makedirs(plots_folder, exist_ok=True)

cls_weight = [1.0 for c in range(n_classes)]
cls_weight = torch.tensor(cls_weight, dtype=torch.float).cuda()
criterion = nn.CrossEntropyLoss(weight=cls_weight, ignore_index=-1).to(device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt['n_epochs'])


train_loss = []
val_loss = []
train_f1 = []
val_f1 = []
train_acc = []
val_acc = []

tolerance = 5
min_delta = 0.3
early_stopping = EarlyStopping(tolerance=tolerance, min_delta=min_delta)
finish = False

print('Running {} #{}'.format(opt['run_name'], opt['run_num']))
min_loss = 10**6
last_train_loss = 10**6

for epoch in range(opt['st_epoch'], opt['st_epoch'] + opt['n_epochs']):
    for phase in ['train', 'validation']:
        running_loss = .0
        y_trues = np.empty([0])
        y_preds = np.empty([0])

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for inputs, labels in datasets[phase]:
            inputs = inputs.to(device)
            labels = labels.long().squeeze().to(device)
            # labels = labels.reshape(-1, 1)

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs).squeeze()

                loss = torch.tensor([0]).to(device)
                calc_bef = False

                for c in range(n_classes):
                    indices = (labels == c).nonzero(as_tuple=False)
                    if indices.numel():
                        real = torch.squeeze(labels[indices], dim=1)
                        predicted = torch.squeeze(outputs[indices], dim=1)
                        if calc_bef:
                            loss += torch.mul(criterion(predicted, real), len(real) / opt['batch_size'])
                        else:
                            loss = torch.mul(criterion(predicted, real), len(real) / opt['batch_size'])
                            calc_bef = True

                if phase == 'train' and calc_bef:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            running_loss += loss.item() * inputs.size(0)

            preds, pred_labels = F.softmax(outputs, 1).max(1)
            
            y_trues = np.append(y_trues, labels.data.cpu().numpy())
            y_preds = np.append(y_preds, labels.data.cpu().numpy())

        epoch_loss = running_loss / len(datasets[phase].sampler)
        acc = accuracy_score(y_trues, y_preds)
        f1 = f1_score(y_trues, y_preds, average='weighted')

        if phase == 'train':
            train_loss.append(epoch_loss)
            train_f1.append(f1)
            train_acc.append(acc)
            last_train_loss = epoch_loss

        else:
            val_loss.append(epoch_loss)
            val_f1.append(f1)
            val_acc.append(acc)
            if epoch_loss < min_loss:
                print('\n\n<<<Saving model>>>\n\n')
                torch.save({'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, new_model_name)
                min_loss = epoch_loss

            early_stopping(last_train_loss, epoch_loss)
            finish = early_stopping.early_stop

        print("[{}] Epoch: {}/{} Loss: {}".format(
            phase, epoch + 1, opt['n_epochs'], epoch_loss), flush=True)
        print('\nF1 Score:\t' + str(f1))
        print('\nAccuracy: \t' + str(acc))

    if finish:
        break


# plotting
plt.plot(range(opt['st_epoch']+1, opt['st_epoch'] + epoch + 2), train_loss, label='train loss')
plt.plot(range(opt['st_epoch']+1, opt['st_epoch'] + epoch + 2), val_loss, label='validation loss')
plt.xlabel('epochs')
plt.ylabel('Losses')
plt.legend()
plt.savefig(join(plots_folder, 'loss.png'))
plt.clf()

plt.plot(range(opt['st_epoch']+1, opt['st_epoch'] + epoch + 2), train_f1, label='train F1')
plt.plot(range(opt['st_epoch']+1, opt['st_epoch'] + epoch + 2), val_f1, label='validation F1')
plt.xlabel('epochs')
plt.ylabel('F1 Scores')
plt.legend()
plt.savefig(join(plots_folder, 'f1_scores.png'))
plt.clf()

plt.plot(range(opt['st_epoch'] + 1, opt['st_epoch'] + epoch + 2), train_acc, label='train accuracy')
plt.plot(range(opt['st_epoch'] + 1, opt['st_epoch'] + epoch + 2), val_acc, label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracies')
plt.legend()
plt.savefig(join(plots_folder, 'accuracy_score.png'))
plt.clf()
