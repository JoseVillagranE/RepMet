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
import glob
import random
import math
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_url

from pathlib import Path
import sys

from utils.debug import set_working_dir
from config.config import config
from utils.download import DownloadGDrive, newDownloadGDrive
import matplotlib.pyplot as plt

class LaSOTDataset(Dataset):
    # setup some class paths
    sub_root_dir = 'LaSOT'
    download_url_prefix = 'http://vision.cs.stonybrook.edu/~lasot/download.html'
    id_drive_all_dataset = "1O2DLxPP8M4Pn4-XCttCJUW3A29tDIeNa"
    id_drive_subset_categories = {
                                "airplane": "1D6xOE5NZ7T8fRYl-ZKcE8R05jXkQ0iev",
                                "basketball": "1Amk8igcqih7yuX5K3shgcSpBIyZyl2zA",
                                "bear": "1wFnqD8ySER0zm28J4e69XP75Vr4F5OzO",
                                "bicycle": "1k3s7YHrjo3tJ838hM_3Dt96yt4JFrEg9",
                                "bird": "1rSghPD62pKlRE2Owd_nqgUmlsY3ZfwBH",
                                "boat": "1iZH5CM1RsMlz1K1-nGEAC5XGNsd3sQ-L",
                                "book": "1rzdSMOQN2rIjzbDhzGs9JiXI2MJBTOyY",
                                "bottle": "19xibe9izagvq9hiEk07JXTw183dOQcht",
                                "bus": "1NBnhcjpa51IOPEkb7paw5SFf8brKqrrK",
                                "car": "1qMNnvnkPQw3-Pg7V3Nud198_F8df1MhI",
                                "cat": "1wzeGBT7kKziGCizuS7j7zbH8A1ckG7EJ",
                                "cattle": "15TbBVFyv78xdoARZ5ijm-XQCX1n_F4f0",
                                "coin": "1vXEO8iIQHtMs2A9f0tPy5nL0Fqwri4R5",
                                "cocodrile": "1NWp2U5a2fHf4siZZJKAsNkHpNUDysKiB",
                                "cup": "17qfzTvRJOIzh_PIrjN3kucLqSX4xW7wW",
                                "dog": "1bDpvh6GPnkhVFv3jag1Kob99D1ItyeSF",
                                "drone": "1HByJQytOmUGFTHgcVhEuwWhgleRi3_yb",
                                "electricfan": "1Z84NQi2fteMA4vEKCkY_rC03-2cMBrVf",
                                "elephant": "1JWJj93kgb6ULJNwNPSgqzGFVuHp5bPFv",
                                "flag": "1Q1eh3v0LBwzp5oIXwTU1Mwf4xDmnGQlV",
                                "fox": "1ODN4M2D5Js8oWdrgA-RYJ0KO0mfS8sgX",
                                "giraffe": "1Ogqs9_THwOS4bnLtfLlhRoIaLniApADv",
                                "gorilla": "1_bTqcYeBrwq_2LK5e-kuw397P1t2NU18",
                                "guitar": "1u2nMBepsUB8POlgoL71Mlb_aG6O1FBdB",
                                "hand": "1ThP7D2ZhQIEzAub5lALbIHVciPfpCs0y",
                                "hat": "1gwIqcZEbhacxxQrAxSaFVwBJQnFI4OOr",
                                "horse": "1dBmHAbizNgzWGLRKi24DQrT4qNILfQzy",
                                "licenseplate": "1xir5p3qrRc7o3-QwxxVP7DOl7zshaPME",
                                "lion": "1OqTt2o0awdVvADCOCAmjFA-4Z6zuNEJM",
                                "monkey": "1AKZ_ula-9DPT7TjkrOgoCQN3iRQ1TJBq",
                                "motorcycle": "12_QZQpCM6-6AOtGDdGZBmjaUimXOSm90",
                                "mouse": "1FtioJXafFTNb_9DEssQy8ljgH_G861-L",
                                "pig": "1FC_UnkbC89f2eMQNUntUZSDchwMKI2FO",
                                "rabbit": "1CGms7-foS0vRB_OX49ZY4NgnwFQLRFVC",
                                "robot": "1MtOt5RzczKyai1aypfiOhv27aNrOMPSi",
                                "rubicCube": "1N-wiyFeLB1U6L8ETZj_htby6HaVw9vI4",
                                "sheep": "10bHTyvns2WoPklJPnm0_wE1kQG-3WjcT",
                                "skateboard": "1dT0tcujrHl3uhSIg9fqIIGXjpttDJ5GX",
                                "spyder": "1qEqJdM8uVhwBIFSp5RAeTiWUXsvR0D6-",
                                "tank": "1eeyMISr3NORpVKRGMR7eDqWatbhDspXG",
                                "tiger": "1jq96zcVwPqb0bJdv-gRkfCGJqsNYRbzO",
                                "train": "1Af0v1NreexXhNNylSAzcDsQJRM6WmqZ-",
                                "truck": "1pD36PhEOo3VhaXB-Ity2pj718At8R2y9",
                                "turtle": "1t90BbodLAvaBx87fgnkSHxLmBOHnXG7q",
                                "umbrella": "1sjq7UXD2Eo79jHgf9rZYTd0trcd5A_QP",
                                "yoyo": "1AjbS2mINIrjfHp81gz-2yjAOFEG_ThCV",
                                "zebra": "1X0FyMR0XQQfZrkCDUL1EwprOYSAVQtC5"

    }
    images_dir = 'Images'

    def __init__(self,
                 root_dir,
                 split='train',
                 crop=True,
                 rate_sample=0,
                 drive=False,
                 transform=None,
                 target_transform=None,
                 force_download=False,
                 categories_subset=None,
                 _sequential_videos=True):
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
        self.rate_sample = rate_sample
        self.transform = transform
        self.target_transform = target_transform
        self.labels = []
        self.drive=drive

        # check if data exists, if not download
        self.download(categories_subset, force=force_download)

        # load the data samples for this split
        # categories(e.x bicycle) and labels (ex. 0)
        self.data, self.labels, self.categories, self.img_filenames, self.gt, \
                self.video_label = self.load_data_split(subcategories=categories_subset, _sequential_videos=_sequential_videos)




        self.samples = list(zip(self.data, self.labels, self.categories, self.img_filenames, self.gt, self.video_label))

        self.n_videos = len(np.unique(self.video_label))
        self.n_categories = len(np.unique(self.labels))

        # create random angles for rotation
        self.angles = [random.uniform(0, 360) for _ in range(len(self.data))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # get the data sample
        sample_data, sample_target, sample_categorie, sample_img_filename, sample_gt, sample_video_label = self.samples[index]
        sample_gt = [int(i) for i in sample_gt]
        box = (sample_gt[0], sample_gt[1], sample_gt[0]+sample_gt[2], sample_gt[1]+sample_gt[3])
        # load the image
        x = self.load_img(join(join(self.root_dir, sample_categorie, sample_video_label, "img"), "%s" % sample_img_filename),
                          crop=self.crop, tuple_crop=box, angle=self.angles[index])

        # perform the transforms
        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, sample_target, sample_categorie

    def download(self, target_obj, force=False):
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

            if self.drive:
                DownloadGDrive(self.id_drive_subset_categories[obj], join(self.root_dir,zip_filename), obj_name=obj)
            else:
                download_url(url, self.root_dir, zip_filename, None)

            zipfile.ZipFile(join(self.root_dir, zip_filename)).extractall(path=join(self.root_dir))
            os.remove(join(self.root_dir, zip_filename))

    def load_data_split(self, subcategories=None, _sequential_videos=True):

        assert self.split in ['train', 'test', 'rand', 'val']

        videos_idxs = []
        if _sequential_videos:
            if self.split == "train":
                videos_idxs = range(15)
            elif self.split == "val":
                videos_idxs = range(15, 18)
            elif self.split == "test":
                videos_idxs = range(15, 20) # I know that each categorie in Lasot have 20 videos
        else:
            random_list = random.sample(range(20), 20)
            if self.split == "train":
                videos_idxs = random_list[:15]
            elif self.split == "val":
                videos_idxs = random_list[15:18]
            elif self.split == "test":
                videos_idxs = random_list[18:]

        # load the samples and their labels
        data = []
        categories = []
        video_label = []
        labels = []
        gt = []
        img_filenames = []
        path = join(self.root_dir)
        categorie_number = 0
        for f_obj in sorted(listdir(path)):
            if subcategories:
                if f_obj in subcategories:
                    for video_idx, video in enumerate(sorted(listdir(join(path, f_obj)))):

                        if video_idx in videos_idxs:
                            # Read GT
                            file = open(os.path.join(path, f_obj, video,'groundtruth.txt'), 'r')
                            data_file = file.readlines()
                            reader = csv.reader(data_file)

                            file_occlusion = open(os.path.join(path, f_obj, video,'full_occlusion.txt'), 'r')
                            data_file_occlusion = file_occlusion.readlines()
                            reader_occlusion = list(csv.reader(data_file_occlusion))[0]

                            file_out_of_view = open(os.path.join(path, f_obj, video,'out_of_view.txt'), 'r')
                            data_out_of_view = file_out_of_view.readlines()
                            reader_out_of_view = list(csv.reader(data_out_of_view))[0]

                            image = Image.open(join(path, f_obj, video, "img", listdir(join(path, f_obj, video, "img"))[0])) # Asume that all video have the same size
                            w_img, h_img = image.size

                            bad_gt_idx = []
                            for idx, row in enumerate(reader):
                                if (idx+1) % self.rate_sample == 0:
                                    if not int(reader_occlusion[idx]) or not int(reader_out_of_view[idx]):
                                        w, h = int(row[2]), int(row[3])
                                        x, y = int(row[0]), int(row[1])
                                        if (w*4 < h or h*4 < w) or  (x < w_img/8 or y < h_img/8 or x > 7*w_img/8 or y > 7*h_img/8):
                                            bad_gt_idx.append(idx)
                                        else:
                                            gt.append(row)
                            # Read Image
                            for idx, img in enumerate(sorted(listdir(join(path, f_obj, video, "img")))):
                                if (idx+1) % self.rate_sample == 0:
                                    if not int(reader_occlusion[idx]) or not int(reader_out_of_view[idx]):
                                        if idx not in bad_gt_idx:
                                            data.append(join(path, f_obj, video, "img", img))
                                            categories.append(f_obj)
                                            video_label.append(video)
                                            labels.append(categorie_number)
                                            img_filenames.append(img)

                    categorie_number += 1

            else:
                for video in listdir(join(path, f_obj)): # obj folder
                    if video_idx in videos_idxs:
                        # Read GT
                        f = open(os.path.join(path, f_obj, video,'groundtruth.txt'), 'r')
                        data_file = f.readlines()
                        reader = csv.reader(data_file)
                        for idx, row in enumerate(reader):
                            gt.append(row)

                        for img in sorted(listdir(join(path, f_obj, f, video, "img"))):
                            data.append(join(path, f_obj, f, video, "img", img))
                            categories.append(f_obj)
                            video_label.append(video)
                            labels.append(categorie_number)
                            img_filenames.append(img)
                categorie_number += 1

        return data, labels, categories, img_filenames, gt, video_label

    @staticmethod
    def load_img(path, crop=False, tuple_crop=None, angle=0):

        # todo either turn image to tensor in transform or do here
        # Load the image
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(path).convert('RGB')
        w_img, h_img = image.size
        if crop:
            if angle > 0:
                # special crop
                w = tuple_crop[2] - tuple_crop[0]
                h = tuple_crop[3] - tuple_crop[1]
                x_center = int(tuple_crop[0] + w/2)
                y_center = int(tuple_crop[1] + h/2)
                r = math.sqrt((w/2)**2 + (h/2)**2)
                r /= math.sqrt(2)
                if (x_center - 1.5*r > 0 or y_center - 1.5*r > 0 or x_center + 1.5*r < w_img or y_center + 1.5*r < h_img) and \
                            (3*w/4 > r or 3*h/4 > r):
                    tuple_crop = (x_center - 1.5*r, y_center - 1.5*r, x_center + 1.5*r, y_center + 1.5*r)
                else:
                    angle = 0 # no rotation
            image = image.crop(tuple_crop)
        if angle > 0:
            image = image.rotate(angle) # angle in degree
            # we need to adjust the image
            w, h = image.size
            x_center, y_center = int(w/2), int(h/2)
            crop = (x_center - r, y_center - r, x_center + r, y_center + r)
            image = image.crop(crop)
        return image

    def stats(self):
        # get the stats to print
        counts = self.class_counts()

        return "%d samples spanning %d classes (avg %d per class)" % \
               (len(self.data), len(counts), int(float(len(self.data))/float(len(counts))))

    def class_counts(self):
        # calculate the number of samples per category
        counts = {}
        for index in range(len(self.samples)):
            sample_data, sample_target, sample_categorie, sample_img_filename, sample_gt, _ = self.samples[index]
            if sample_target not in counts:
                counts[sample_target] = 1
            else:
                counts[sample_target] += 1

        return counts


if __name__ == "__main__":


    set_working_dir()

    target_obj = ["airplane"]
    # load the dataset
    # Lasot -> 30fps
    dataset = LaSOTDataset(root_dir=config.dataset.root_dir, rate_sample=30, categories_subset=target_obj, drive=True, split='train')

    # print the stats
    print(dataset.stats())

    # lets plot some samples
    fig = plt.figure()
    j = 0
    for i in range(340, len(dataset)):
        sample = dataset.__getitem__(i)

        ax = plt.subplot(1, 4, j + 1)
        plt.tight_layout()
        # ax.set_title('Sample %d - Class %s - Categorie %s' % (i, sample[2], sample[1]))  # convert label to categ.
        ax.axis('off')
        plt.imshow(sample[0])  # todo when tensor will need to convert tensor to img
        j += 1
        if j == 3:
            plt.show()
            break
