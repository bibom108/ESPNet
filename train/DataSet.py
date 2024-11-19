import torch
import cv2
import torch.utils.data
import numpy as np

__author__ = "Sachin Mehta"


class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, imList, labelList, transform=None):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''
        self.imList = imList
        self.labelList = labelList
        self.transform = transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image = cv2.imread(image_name)
        # label = cv2.imread(label_name, 0)
        label = cv2.imread(label_name)
        
        label = label[:, :, 0]
        # label = convert_to_binary_mask(label)
        
        # if self.labelList[idx] is None:
        #     print("="*50)
        # print(self.labelList[idx], label.shape)
        # exit()
        if self.transform:
            [image, label] = self.transform(image, label)
        return (image, label)


def convert_to_binary_mask(mask):
    non_road_label = np.array([255, 0, 0])
    road_label = np.array([255, 0, 255])
    other_road_label = np.array([0, 0, 0])

    binary_mask = np.all(mask == road_label, axis=2).astype(np.uint8)

    binary_mask = np.expand_dims(binary_mask, axis=-1)
    return binary_mask