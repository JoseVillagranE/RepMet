# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 23:21:47 2020

@author: joser


LaSot dataset loader

https://cis.temple.edu/lasot

"""

from __future__ import print_function

from PIL import Image, ImageFile
from os.path import join
import os
from os import listdir
import numpy as np
import scipy.io
import tarfile
import zipfile
import shutil
import csv
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_url

from pathlib import Path
import sys

from utils.debug import set_working_dir
from config.config import config
from utils.download import DownloadGDrive
import matplotlib.pyplot as plt

class LaSOTDataset(Dataset):
    # setup some class paths
    sub_root_dir = 'LaSOT'
    download_url_prefix = 'https://cis.temple.edu/lasot/data/category/'
    id_drive = '1v09JELSXM_v7u3dF7akuqqkVG8T1EK2_'
    images_dir = 'Images'

    def __init__(self,
                 root_dir,
                 target_obj,
                 split='train',
                 crop=True,
                 drive=False,
                 transform=None,
                 target_transform=None,
                 force_download=False,
                 categories_subset=None):
        """
        :param root_dir: (string) the directory where the dataset will be stored
        :param split: (string) 'train', 'trainval', 'val' or 'test'
        :param transform: how to transform the input
        :param target_transform: how to transform the target
        :param force_download: (boolean) force a new download of the dataset
        :param categories_subset: (iterable) specify a subset of categories to build this set from
        """

        super().__init__()

        # set instance variables
        self.root_dir = join(os.path.expanduser(root_dir), self.sub_root_dir)
        self.split = split
        self.crop = crop
        self.transform = transform
        self.target_transform = target_transform
        self.labels = []

        # check if data exists, if not download
        self.download(target_obj, force=force_download)

        # load the data samples for this split
        # categories(e.x bicycle) and labels (ex. bicycle_1, bicycle_2 ...)
        self.data, self.labels, self.categories, self.gt = self.load_data_split(categories_subset=categories_subset)
        self.samples = list(zip(self.data, self.categories, self.labels, self.gt))

        self.n_videos = len(np.unique(self.labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # get the data sample
        sample_data, sample_target, sample_categorie, sample_gt = self.samples[index]

        box = (sample_gt[0], sample_gt[1], sample_gt[0]+sample_gt[2], sample_gt[1]+sample_gt[3])

        # load the image
        x = self.load_img(join(join(self.root_dir, sample_categorie, sample_target, "img"), "%s" % sample_data),
                          crop=self.crop, tuple_crop=box)
        
        y = sample_categorie # Cuidado. Lo más posible es que deba modificar

        # perform the transforms
        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def download(self, target_obj, force=False, drive=False):
        """
        

        Parameters
        ----------
        target_obj : .txt
                    objects to work with
        force : Boolean, optional.

        Returns
        -------
        None.

        """
        # check for existence, if so return
        for obj in target_obj:
            if os.path.exists(join(self.root_dir, obj)):
                if not force:
                    print('File {} already downloaded and verified'.format(obj))
                    return
                else:
                    shutil.rmtree(join(self.root_dir, obj))
    
            # make the dirs and start the downloads
            os.makedirs(self.root_dir, exist_ok=True)
            filename = obj
            zip_filename = filename + '.zip'
            url = join(self.download_url_prefix, zip_filename)
            
            if drive:
                DownloadGDrive(self.id_drive, self.root_dir)
            else:
                download_url(url, self.root_dir, zip_filename, None)
            
            zipfile.ZipFile.extractall(join(self.root_dir, zip_filename))
            os.remove(join(self.root_dir, zip_filename))

    def load_data_split(self, subcategories=None):

        assert self.split in ['train', 'test', 'rand', 'val']

        # load the samples and their labels
        data = []
        categories = []
        labels = []
        gt = []
        path = join(self.root_dir)
        for f_obj in listdir(path):
            if subcategories:
                categories.append(f_obj)
                if f_obj in subcategories:
                    for f in listdir(join(path, f_obj)):
                        labels.append(f)
                        # Read GT
                        f = open(os.path.join(path, f_obj, f,'groundtruth.txt'), 'r')
                        data = f.readlines()
                        reader = csv.reader(data)
                        for idx, row in enumerate(reader):
                            gt.append(row)
                        # Read Image
                        for img in listdir(join(path, f_obj, f, "img")): 
                            data.append(join(path, f_obj, f, "img", img))
                        
            else:
                categories.append(f_obj)
                for f in listdir(join(path, f_obj)):
                    labels.append(f)
                    
                    # Read GT
                    f = open(os.path.join(path, f_obj, f,'groundtruth.txt'), 'r')
                    data = f.readlines()
                    reader = csv.reader(data)
                    for idx, row in enumerate(reader):
                        gt.append(row)
                    
                    for img in listdir(join(path, f_obj, f, "img")):
                        data.append(join(path, f_obj, f, "img", img))

        return data, labels, categories, gt

    @staticmethod
    def load_img(path, crop=False, tuple_crop=None, rot=False, angle=0):

        # todo either turn image to tensor in transform or do here
        # Load the image
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(path).convert('RGB')
        
        if crop:
            image = image.crop(tuple_crop)
        
        if rot:
            image = image.rotate(angle) # angle in degree


        return image

    def stats(self):
        # get the stats to print
        counts = self.class_counts()

        return "%d samples spanning %d classes (avg %d per class)" % \
               (len(self.samples), len(counts), int(float(len(self.samples))/float(len(counts))))

    def class_counts(self):
        # calculate the number of samples per category
        counts = {}
        for index in range(len(self.samples)):
            sample_data, sample_target = self.samples[index]
            if sample_target not in counts:
                counts[sample_target] = 1
            else:
                counts[sample_target] += 1

        return counts


if __name__ == "__main__":
    

    set_working_dir()
    
    target_obj = ["coin"]
    # load the dataset
    dataset = LaSOTDataset(root_dir=config.dataset.root_dir, target_obj=target_obj, drive=True, split='test')

    # print the stats
    print(dataset.stats())

    # lets plot some samples
    fig = plt.figure()
    
    ex = dataset.__getitem__(0)
    
    for i in range(len(dataset)):
        sample = dataset[i]

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample %d - Class %s' % (i, sample[1]))  # convert label to categ.
        ax.axis('off')
        plt.imshow(sample[0])  # todo when tensor will need to convert tensor to img

        if i == 3:
            plt.show()
            break