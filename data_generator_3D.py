import torch
import numpy as np
import time
import math
import os
import random
import nibabel as nib

base_path = '.../data/train_valid_test/'  # 改成你的路径
size_x = 240
size_y = 160
size_z = 48
class Covid19TrainSet():
    def __iter__(self):
        file = "/home/ubuntu/zhaoqianfei/data/train_valid_test/config/image_train_names.txt"
        train_list = []
        with open(file) as f:
            for line in f:
                for i in line.split():
                    train_list.append(int(i))

        for i in train_list:
            image = nib.load(base_path + 'image/' + str(i) + '.nii.gz')
            image = np.asarray(image.dataobj)[np.newaxis, np.newaxis, :,  :, :]
            label = nib.load(base_path + 'label/' + str(i) + '.nii.gz')
            label = np.asarray(label.dataobj)[np.newaxis, np.newaxis, :,  :, :]
            x = image.shape[2]
            y = image.shape[3]
            z = image.shape[4]
            x_random = random.randrange(0, x-size_x)
            y_random = random.randrange(0, y-size_y)
            z_random = random.randrange(0, z-size_z) if z > 64 else 0
            image_random = image[:,:, x_random:x_random+size_x, y_random:y_random+size_y, z_random:z_random+size_z]
            label_random = label[:,:, x_random:x_random+size_x, y_random:y_random+size_y, z_random:z_random+size_z]

            yield str(i) + '.nii.gz', image_random, label_random

        return

    def __len__(self):
        return 80


class Covid19EvalSet():
    def __iter__(self):
        file = ".../data/train_valid_test/config/image_valid_names.txt"
        train_list = []
        with open(file) as f:
            for line in f:
                for i in line.split():
                    train_list.append(int(i))

        for i in train_list:
            image = nib.load(base_path + 'image/' + str(i) + '.nii.gz')
            image = np.asarray(image.dataobj)[np.newaxis, np.newaxis, :,  :, :]
            label = nib.load(base_path + 'label/' + str(i) + '.nii.gz')
            label = np.asarray(label.dataobj)[np.newaxis, np.newaxis, :,  :, :]
            z = image.shape[4]
            z_random = random.randrange(0, z-size_z) if z > 64 else 0
            image_random = image[:,:, :, :, z_random:z_random+size_z]
            label_random = label[:,:, :, :, z_random:z_random+size_z]
            yield str(i) + '.nii.gz', image_random, label_random
        return

    def __len__(self):
        return 13


class Convid19TestSet:
    def __iter__(self):
        file = ".../data/train_valid_test/config/image_test_names.txt"
        train_list = []
        with open(file) as f:
            for line in f:
                for i in line.split():
                    train_list.append(int(i))
        #train_list = [31]
        for i in train_list:
            image = nib.load(base_path + 'image/' + str(i) + '.nii.gz')
            image = np.asarray(image.dataobj)[np.newaxis, np.newaxis, :,  :, :]
            label = nib.load(base_path + 'label/' + str(i) + '.nii.gz')
            label = np.asarray(label.dataobj)[np.newaxis, np.newaxis, :,  :, :]

            yield str(i) + '.nii.gz', image, label

        return

'''train_loader = Covid19TrainSet()
for step, (name, X, y) in enumerate(train_loader):
    print("???")'''
