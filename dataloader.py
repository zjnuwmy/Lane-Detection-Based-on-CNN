from __future__ import print_function, division
import os
import torch
import random as rn
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms

import torchvision.transforms.functional as F


import pandas as pd
from PIL import Image

import utils as utils

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from functions import raw_to_cnn
from img_serializer import deserialize_image
class SimulationDataset(Dataset):
    """Dataset wrapping input and target tensors for the driving simulation dataset.

    Arguments:
        set (String):  Dataset - train, test
        path (String): Path to the csv file with the image paths and the target values
    """
    def get_servo_dataset(filename, start_index=0, end_index=None, conf='./settings.json'):
    #    data = pd.DataFrame.from_csv(filename,encoding='utf8')
        #data = pd.read_csv(filename,encoding='utf8',engine='python',error_bad_lines=False)
        data = pd.read_csv(filename)

        # Outputs
        x = []

        # Servo ranges from 40-150
        servo = []

        for i in data.index[start_index:end_index]:
            # Don't want noisy data
    #        if data['servo'][i] < 40 or data['servo'][i] > 150:
    #            continue

            # Append
            x.append(deserialize_image(data['image'][i], config=conf))
            servo.append(raw_to_cnn(data['servo'][i]))

        return x, servo

    #DATA_FILES = ['C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_0.csv', 'C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_1.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_2.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_3.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_4.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_5.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_6.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_7.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_8.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_9.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_10.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_11.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_12.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_13.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_14.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_15.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_16.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_17.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_18.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_19.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_20.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_21.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_22.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_23.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_24.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_25.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_26.csv']
    '''
    csv_path = "C:/Users/circle/Desktop/RCDATA_CSV_7_15"
    dir_len = len(os.listdir(csv_path))
    for i in range(dir_len):
        DATA_FILES = 'C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_%d.csv'%i
    '''
    DATA_FILES = ['C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_{}.csv'.format(i) for  i in range(27)]
    def __init__(self,DATA_FILES):
        
        
        #self.data = pd.read_csv(csv_path, error_bad_lines=False)
        # First column contains the middle image paths
        # Fourth column contains the steering angle

#        self.image_paths = np.array(self.data.iloc[start:end, 0:3])
        print('[!] Loading dataset...')
        X = []
        SERVO = []
        
        for d in DATA_FILES:
            c_x, c_servo = SimulationDataset.get_servo_dataset(d)
            X = X + c_x
            SERVO = SERVO + c_servo

        X = np.array(X)
        SERVO = np.array(SERVO) 
        print('[!] Finished loading dataset...')
        
        self.images = X
        self.targets = SERVO

        # Preprocess and filter data
        self.targets = gaussian_filter1d(self.targets, 2)
        # bias = 0.03
        # self.image_paths = [image_path for image_path, target in zip(self.image_paths, self.targets) if
        #                     abs(target) > bias]
        # self.targets = [target for target in self.targets if abs(target) > bias]
        
    def __getitem__(self, index):
        # Get image name from the pandas df
        #image_paths = self.images[index]
        # Open image
        #image= np.fromstring(self.images[index][1:-1], dtype=int, sep=', ')
#        print(len(image))
        #im = np.reshape(images, (40, 70, 3))
#        print(im)
        #image= Image.fromarray(im.astype('uint8'))
        #image = np.array(image)
        # plt.imshow(image)
        # plt.show()

    # image = [Image.open(image_paths[i]) for i in range(3)]
        image = self.images[index]

        target = self.targets[index]

        sample = {'image': image, 'target': target}
        #print("before sample values:",sample["image"])
        
        sample["image"] = F.to_tensor(sample["image"])
        sample["target"] = torch.tensor(float(target))
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        '''
        if self.transforms is not None:
            sample["image"] = self.transforms(sample["image"])
            target = sample["target"]
            sample["target"] = torch.tensor(float(target))
        '''

        # plt.imshow(F.to_pil_image(sample['image']))

        # plt.title(str(sample['target']))
        # plt.show()
        #sample = {'image': image, 'target': target}
        #print("last sample values:",sample["image"])
        #print("last label values:",sample["target"])
        return sample['image'], sample['target']

    def __len__(self):
        return len(self.images)

