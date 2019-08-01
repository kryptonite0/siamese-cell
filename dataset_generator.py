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

def get_image_path(basepath_data, original_image_size, img_id):
    image_path_train = basepath_data + f"resized_{original_image_size}/" + "train/" + f"{img_id}.npy"
    if os.path.exists(image_path_train):
        return image_path_train
    else:
        image_path_test = basepath_data + f"resized_{original_image_size}/" + "test/" + f"{img_id}.npy"
        if os.path.exists(image_path_test):
            return image_path_test
        else:
            raise RuntimeError(f"Image path {image_path_train} or {image_path_test} not found!")

class RXRXDataset(torchDataset):
    """
        RXRX dataset for siRNA classification. 
        Each element of the dataset is a 6-channel image of a site of a well.
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
        
    def __getitem__(self, index1):
        """
        :param index:
        :return: (tuple) (image1, image2, is_same_label)
        """
        # generate random 0-1
        is_same_label = random.randint(0,1)
        # get img_ids
        img_id1 = self.id_list[index1]
        label1 = self.label_list[index1]
        if is_same_label:
            id_list_same = self.id_list[self.label_list==label1]
            img_id2 = img_id1
            while img_id2 == img_id1:
                img_id2 = random.choice(id_list_same)
            label2 = label1
        else:
            id_list_diff = self.id_list[self.label_list!=label1]
            img_id2 = random.choice(id_list_diff)
            index2 = np.where(self.id_list == img_id2)[0][0]
            label2 = self.label_list[index2]
            
        # load 6-channel images from .npy file
        img_path1 = get_image_path(self.basepath_data, self.original_image_size, img_id1)
        img_path2 = get_image_path(self.basepath_data, self.original_image_size, img_id2)
        img1 = np.load(img_path1).astype('float64')
        img2 = np.load(img_path2).astype('float64')
        # get normalization coefficients for its experiment
        exp1, exp2 = img_id1.split("_")[0], img_id2.split("_")[0]
        norm1, norm2 = self.exp_norm_dict[exp1], self.exp_norm_dict[exp2]
        # normalize
        img1 -= norm1["median"] 
        img1 /= norm1["std"]
        img2 -= norm2["median"] 
        img2 /= norm2["std"]
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, is_same_label, str(label1), str(label2)

    def __len__(self):
        return len(self.id_list)

def create_datasets_and_loaders(data, batch_size, basepath_data, original_image_size):
    
    transform_train = tv.transforms.Compose([tv.transforms.ToTensor()])
    transform_valid = tv.transforms.Compose([tv.transforms.ToTensor()])
    transform_test = tv.transforms.Compose([tv.transforms.ToTensor()])
    
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
    dataset_valid_debug = RXRXDataset(id_list=data["ids_valid_debug"], label_list=data["labels_valid_debug"],
                                      basepath_data=basepath_data, original_image_size=original_image_size, 
                                      transform=transform_valid, exp_norm_dict=data["exp_norm_dict"])
    # dataset_test = RXRXDataset(id_list=data["ids_test"], label_list=data["labels_test"],
    #                            basepath_data=basepath_data, original_image_size=original_image_size, 
    #                            transform=transform_test, exp_norm_dict=data["exp_norm_dict"])
    # dataset_test_debug = RXRXDataset(id_list=data["ids_test"][:8], label_list=data["labels_test"][:8],
    #                                  basepath_data=basepath_data, original_image_size=original_image_size, 
    #                                  transform=transform_test, exp_norm_dict=data["exp_norm_dict"])

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
    # loader_test = DataLoader(dataset=dataset_test,
    #                          batch_size=batch_size,
    #                          shuffle=False,
    #                          pin_memory=True,
    #                          num_workers=multiprocessing.cpu_count()) 
    # loader_test_debug = DataLoader(dataset=dataset_test_debug,
    #                                batch_size=batch_size,
    #                                shuffle=False,
    #                                pin_memory=True,
    #                                num_workers=multiprocessing.cpu_count()) 

    # store datasets and dataloader in dictionaries
    datasets = {"train" : dataset_train, "train_debug" : dataset_train_debug,
                "valid" : dataset_valid, "valid_debug" : dataset_valid_debug,
                # "test" : dataset_test,   "test_debug" : dataset_test_debug
               }
    loaders = {"train" : loader_train, "train_debug" : loader_train_debug,
               "valid" : loader_valid, "valid_debug" : loader_valid_debug,
               # "test" : loader_test,   "test_debug" : loader_test_debug
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

def show_batch(loader):
    img_batch1, img_batch2, is_same_batch, label_batch1, label_batch2 = next(iter(loader))
    print("Image batch size: {}. Label batch size: {}.".format(img_batch1.size(), len(label_batch1)))
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
                label1, label2 = label_batch1[k], label_batch2[k]
                is_same = is_same_batch[k]
                axs[2*i,j].imshow(rio.convert_tensor_to_rgb(img1))
                axs[2*i+1,j].imshow(rio.convert_tensor_to_rgb(img2))
                color = "green" if is_same else "red"
                axs[2*i,j].set_title(label1, color=color)
                axs[2*i+1,j].set_title(label2, color=color)
    
    plt.show()
