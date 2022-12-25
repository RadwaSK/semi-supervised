import os
import shutil
from os.path import join


testing_labels = open('dataset/test.txt', 'r').readlines()
data_path = 'dataset'

for t in testing_labels:
    path, label = t.split(' ')
    label = label.strip('\n\t ')
    src = join(data_path, path)
    path = path.split('/')
    # print(label)
    dst = join(data_path, path[0], str(label))
    os.makedirs(dst, exist_ok=True)
    dst = join(dst, path[1])
    print('copying:', path[1])
    shutil.move(src, dst)


data = 'dataset/test'
data = os.listdir(data)

for d in data:
    if d.find('.') == -1:
        path = join('dataset/test', d)
        if len(os.listdir(path)) == 0:
            shutil.rmtree(path)
