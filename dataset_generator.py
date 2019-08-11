# import standard libs
from torch.utils.data.dataset import Dataset as torchDataset
import torchvision as tv
from torch.utils.data import DataLoader

import numpy as np
import random
import os
import multiprocessing
from matplotlib import pyplot as plt
import warnings

import sys
sys.path.append("/jet/prs/workspace/rxrx1-utils")
from rxrx import io as rio

from albumentations import Compose, RandomCrop, Rotate, HorizontalFlip, VerticalFlip, Resize
from albumentations.pytorch import ToTensor

def get_image_path(basepath_data, original_image_size, img_id):
    dataset = img_id.split("_")[0]
    img_id = "_".join(img_id.split("_")[1:])
    image_path = os.path.join(basepath_data, f"resized_{original_image_size}", 
                              f"{dataset}", f"{img_id}.npy")
    if not os.path.exists(image_path):
        raise RuntimeError(f"Image {image_path} not found!")
    
    return image_path

class SiameseRXRXDataset(torchDataset):
    """
        RXRX dataset for siamese siRNA verification. 
        Each element of the dataset is a pair of 6-channel images from the same class; 
        each image corresponds to a site of a well.
        https://www.rxrx.ai/
    """

    def __init__(self, id_list, label_list, basepath_data, 
                 original_image_size, transform, exp_norm_dict):
        """
        :param id_list: (list of str) list of unique image IDs
        :param label_list: (list of str) list of labels
        :param basepath_data: (str) basepath of data
        :param original_image_size: (int) original image size (image shape is supposed to be square)
        :param transform: (callable, optional) optional transform to be applied on a sample
        :param exp_norm_dict: (dict) dictionary of experiment -> mean_median_pixel_value_by_channel, 
                              mean_std_pixel_value_by_channel used for normalization of images
        """

        # initialize variables
        self.id_list = np.array(id_list)
        self.label_list = np.array(label_list)
        self.basepath_data = basepath_data
        self.original_image_size = original_image_size
        self.transform = transform
        self.exp_norm_dict = exp_norm_dict
        
    def __getitem__(self, index1):
        """
        :param index:
        :return: (tuple) (image1, image2)
        """
        # get img_ids
        img_id1 = self.id_list[index1]
        label = self.label_list[index1]
        id_list_same = self.id_list[self.label_list==label]
        # ensures the pair is never composed by the same image twice
        img_id2 = img_id1
        while img_id2 == img_id1:
            img_id2 = random.choice(id_list_same)
            
        # load 6-channel images from .npy file
        img_path1 = get_image_path(self.basepath_data, self.original_image_size, img_id1)
        img_path2 = get_image_path(self.basepath_data, self.original_image_size, img_id2)
        img1 = np.load(img_path1).astype('float64')
        img2 = np.load(img_path2).astype('float64')
        # get normalization coefficients for its experiment
        exp1, exp2 = img_id1.split("_")[1], img_id2.split("_")[1]
        norm1, norm2 = self.exp_norm_dict[exp1], self.exp_norm_dict[exp2]
        # normalize
        img1 -= norm1["mean"] 
        img1 /= norm1["std"]
        img2 -= norm2["mean"] 
        img2 /= norm2["std"]
        # apply augmentation
        if self.transform:
            img1 = self.transform(image=img1)['image']
            img2 = self.transform(image=img2)['image']
                
        return img1, img2, str(label)

    def __len__(self):
        return len(self.id_list)

def create_siamese_datasets_and_loaders(data, batch_size, basepath_data, original_image_size):
    
    transform_train = Compose([HorizontalFlip(p=0.5),
                               VerticalFlip(p=0.5),
                               Rotate(limit=180, p=1),
                               # RandomCrop(int(0.66*original_image_size), int(0.66*original_image_size)),
                               # Resize(original_image_size, original_image_size),
                               ToTensor()
                            ])

    transform_valid = Compose([ToTensor()])
    transform_test = Compose([ToTensor()])
    
    # create datasets
    dataset_train = SiameseRXRXDataset(id_list=data["ids_train"], label_list=data["labels_train"],
                                        basepath_data=basepath_data, original_image_size=original_image_size, 
                                        transform=transform_train, exp_norm_dict=data["exp_norm_dict"])
    dataset_train_debug = SiameseRXRXDataset(id_list=data["ids_train_debug"], label_list=data["labels_train_debug"],
                                              basepath_data=basepath_data, original_image_size=original_image_size, 
                                              transform=transform_train, exp_norm_dict=data["exp_norm_dict"])
    dataset_valid = SiameseRXRXDataset(id_list=data["ids_valid"], label_list=data["labels_valid"],
                                        basepath_data=basepath_data, original_image_size=original_image_size, 
                                        transform=transform_valid, exp_norm_dict=data["exp_norm_dict"])
    dataset_valid_debug = SiameseRXRXDataset(id_list=data["ids_valid_debug"], label_list=data["labels_valid_debug"],
                                              basepath_data=basepath_data, original_image_size=original_image_size, 
                                              transform=transform_valid, exp_norm_dict=data["exp_norm_dict"])
    
    # create the dataloaders
    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=multiprocessing.cpu_count()) 
    loader_train_debug = DataLoader(dataset=dataset_train_debug,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=multiprocessing.cpu_count()) 
    loader_valid = DataLoader(dataset=dataset_valid,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=multiprocessing.cpu_count()) 
    loader_valid_debug = DataLoader(dataset=dataset_valid_debug,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=multiprocessing.cpu_count()) 

    # store datasets and dataloader in dictionaries
    datasets = {"train" : dataset_train, "train_debug" : dataset_train_debug,
                "valid" : dataset_valid, "valid_debug" : dataset_valid_debug,
               }
    loaders = {"train" : loader_train, "train_debug" : loader_train_debug,
               "valid" : loader_valid, "valid_debug" : loader_valid_debug,
              }
    
    return datasets, loaders

# define a MinMaxScaler function for the images
def imgMinMaxScaler(img, scale_range=(0,255), dtype='uint8'):
    """
    :param img: image to be rescaled
    :param scale_range: (tuple) (min, max) of the desired rescaling
    """
    warnings.filterwarnings("ignore")
    img = img.astype('float64')
    img_std = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_scaled = img_std * float(scale_range[1] - scale_range[0]) + float(scale_range[0])
    # round at closest integer and transform to integer 
    img_scaled = np.rint(img_scaled).astype(dtype)

    return img_scaled

def show_siamese_batch(loader):
    img_batch1, img_batch2, label_batch = next(iter(loader))
    print("Image batch size: {}. Label batch size: {}.".format(img_batch1.size(), len(label_batch)))
    print("Average pixel value in batch: {:.5f}".format((img_batch1.mean() + img_batch1.mean())/2.))
    print("Stddev pixel value in batch: {:.5f}".format((img_batch1.std() + img_batch2.std())/2.))
    # display batch
    n_cols = 8
    n_rows = 2 * (img_batch1.size()[0] // n_cols + 1*(img_batch1.size()[0]%n_cols>0))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*15/n_cols))
    for i in range(n_rows//2):
        for j in range(n_cols):
            k = i * n_cols + j
            if k<img_batch1.size()[0]:    
                img1 = img_batch1[k].permute(1,2,0).data.cpu().numpy()
                img2 = img_batch2[k].permute(1,2,0).data.cpu().numpy()
                img1, img2 = imgMinMaxScaler(img1), imgMinMaxScaler(img2)
                label = label_batch[k]
                axs[2*i,j].imshow(rio.convert_tensor_to_rgb(img1))
                axs[2*i+1,j].imshow(rio.convert_tensor_to_rgb(img2))
                axs[2*i,j].set_title(label)
                axs[2*i+1,j].set_title(label)
    
    plt.show()

class RXRXDataset(torchDataset):
    """
        RXRX dataset for testing siRNA verification. 
        Each element of the dataset is a 6-channel image; 
        each image corresponds to a site of a well.
        https://www.rxrx.ai/
    """

    def __init__(self, id_list, label_list, basepath_data, 
                 original_image_size, transform, exp_norm_dict):
        """
        :param id_list: (list of str) list of unique image IDs
        :param label_list: (list of str) list of labels
        :param basepath_data: (str) basepath of data
        :param original_image_size: (int) original image size (image shape is supposed to be square)
        :param transform: (callable, optional) optional transform to be applied on a sample
        :param exp_norm_dict: (dict) dictionary of experiment -> mean_median_pixel_value_by_channel, 
                              mean_std_pixel_value_by_channel used for normalization of images
        """

        # initialize variables
        self.id_list = np.asarray(id_list)
        self.label_list = np.asarray(label_list)
        self.basepath_data = basepath_data
        self.original_image_size = original_image_size
        self.transform = transform
        self.exp_norm_dict = exp_norm_dict
        
    def __getitem__(self, index):
        """
        :param index:
        :return: (tuple) (image2, label)
        """
        # get img_id
        img_id = self.id_list[index]
        label = self.label_list[index]            
        # load 6-channel image from .npy file
        img_path = get_image_path(self.basepath_data, self.original_image_size, img_id)
        img = np.load(img_path).astype('float64')
        # get normalization coefficients for its experiment
        exp = img_id.split("_")[1]
        norm = self.exp_norm_dict[exp]
        # normalize
        img -= norm["mean"] 
        img /= norm["std"]
        # apply augmentation
        if self.transform:
            img = self.transform(image=img)['image']
                
        return img, str(label)

    def __len__(self):
        return len(self.id_list)

def create_predict_datasets_and_loaders(data, batch_size, basepath_data, original_image_size):
    
    transform_train = Compose([HorizontalFlip(p=0.5),
                               VerticalFlip(p=0.5),
                               Rotate(limit=180, p=1),
                               # RandomCrop(int(0.66*original_image_size), int(0.66*original_image_size)),
                               # Resize(original_image_size, original_image_size),
                               ToTensor()
                            ])
    # transform_train = Compose([ToTensor()])
    transform_valid = Compose([ToTensor()])
    transform_test = Compose([ToTensor()])
    
    # create datasets
    dataset_train = RXRXDataset(id_list=data["ids_train"], label_list=data["labels_train"],
                                basepath_data=basepath_data, original_image_size=original_image_size, 
                                transform=transform_train, exp_norm_dict=data["exp_norm_dict"])
    dataset_train_debug = RXRXDataset(id_list=data["ids_train_debug"], label_list=data["labels_train_debug"],
                                      basepath_data=basepath_data, original_image_size=original_image_size, 
                                      transform=transform_train, exp_norm_dict=data["exp_norm_dict"])
    dataset_valid = RXRXDataset(id_list=data["ids_valid"], label_list=data["labels_valid"],
                                basepath_data=basepath_data, original_image_size=original_image_size, 
                                transform=transform_valid, exp_norm_dict=data["exp_norm_dict"])
    dataset_test = RXRXDataset(id_list=data["ids_test"], label_list=data["labels_test"],
                               basepath_data=basepath_data, original_image_size=original_image_size, 
                               transform=transform_valid, exp_norm_dict=data["exp_norm_dict"])
    
    # create the dataloaders
    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=multiprocessing.cpu_count()) 
    loader_train_debug = DataLoader(dataset=dataset_train_debug,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=multiprocessing.cpu_count()) 
    loader_valid = DataLoader(dataset=dataset_valid,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=multiprocessing.cpu_count()) 
    loader_test = DataLoader(dataset=dataset_test,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=multiprocessing.cpu_count()) 

    # store datasets and dataloader in dictionaries
    datasets = {"train" : dataset_train, "train_debug" : dataset_train_debug, 
                "valid" : dataset_valid, 
                "test" : dataset_test,
               }
    loaders = {"train" : loader_train, "train_debug" : loader_train_debug, 
               "valid" : loader_valid, 
               "test" : loader_test,
              }
    
    return datasets, loaders

def show_predict_batch(loader):
    img_batch, label_batch = next(iter(loader))
    print("Image batch size: {}. Label batch size: {}.".format(img_batch.size(), len(label_batch)))
    print("Average pixel value in batch: {:.5f}".format((img_batch.mean() + img_batch.mean())/2.))
    print("Stddev pixel value in batch: {:.5f}".format((img_batch.std() + img_batch.std())/2.))
    # display batch
    n_cols = 8
    n_rows = (img_batch.size()[0] // n_cols + 1*(img_batch.size()[0]%n_cols>0))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*15/n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            k = i * n_cols + j
            if k<img_batch.size()[0]:    
                img = img_batch[k].permute(1,2,0).data.cpu().numpy()
                img = imgMinMaxScaler(img)
                label = label_batch[k]
                axs[i,j].imshow(rio.convert_tensor_to_rgb(img))
                axs[i,j].set_title(label)
    
    plt.show()
