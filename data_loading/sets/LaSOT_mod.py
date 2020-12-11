from PIL import Image, ImageFile
import glob
import os
import csv
from os import listdir
import numpy as np
import random
import zipfile
from config.config import config
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_url
from utils.download import DownloadGDrive



class LaSOTDataset_mod(Dataset):

    sub_root_dir = 'LaSOT-modified'
    id_drive_subset_categories = {
                                    "book": "1bGaTImz9hOwIdcljNvo_JQoVkUOIe0zi",
                                    "bottle": "1j0oULJFkLiSiVacKKFuJbNDFEsCThphK",
                                    "coin": "1-Iw0dyoW1ZjfM2NfErjD-0kduDPqOdKp",
                                    "cup": "1t5J9eegIrygUG9iRlzrV98Y_fGjbL9Gp",
                                    "robot": "1-3FJawcSETT6opcvX9_AqQKYR0t4Pule",
                                    "rubicCube": "1YSdTO1w0aaEtLSaUcHSohUbYgApHeQao"
                                    }

    def __init__(self,
                 root_dir,
                 idxs_videos,
                 num_videos,
                 split='train',
                 crop=True,
                 transform=None,
                 angle_in_rad=True,
                 rotate_image=False,
                 force_download=False,
                 objects = ["book", "bottle", "coin", "cup", "robot", "rubicCube"]):

        self.root_dir = os.path.join(os.path.expanduser(root_dir), self.sub_root_dir)
        self.idxs_videos = idxs_videos
        self.num_videos = num_videos
        self.split = split
        self.crop = crop
        self.transform = transform
        self.angle_in_rad = angle_in_rad
        self.rotate_image = rotate_image

        # check if data exists, if not download
        self.download(objects, force=force_download)

        self.data, self.gt, self.labels, self.categories = self.load_data_split(self.root_dir,
                                                                                split,
                                                                                idxs_videos,
                                                                                num_videos)

        assert len(self.data) == len(self.gt), f"len_data: {len(self.data)}, len_gt: {len(self.gt)}"

        self.samples = list(zip(self.data, self.gt, self.labels, self.categories))
        self.n_categories = len(np.unique(self.labels))

        if rotate_image:
            # create random angles for rotation
            self.angles = [random.uniform(0, 360) for _ in range(len(self.data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data, gt, label, categorie = self.samples[index]
        sample_gt = [int(i) for i in gt]
        box = (sample_gt[0], sample_gt[1], sample_gt[0]+sample_gt[2], sample_gt[1]+sample_gt[3])

        rotation_angle = 0
        if self.rotate_image:
            x, original_image = self.load_img(data, crop=self.crop, tuple_crop=box, angle=self.angles[index])
            rotation_angle = self.angles[index]

        else:
            x, original_image = self.load_img(data, crop=self.crop, tuple_crop=box, angle=0)

        if self.transform:
            x = self.transform(x)
            original_image = self.transform(original_image)

        if self.angle_in_rad:
            rotation_angle = self.degree2rad(rotation_angle)

        return x, label, categorie, rotation_angle, original_image


    def download(self, target_obj, force=False):

        # check for existence, if so return
        for obj in target_obj:
            if os.path.exists(os.path.join(self.root_dir, obj)):
                if not force:
                    print('File {} already downloaded and verified'.format(obj))
                    return
                else:
                    shutil.rmtree(join(self.root_dir, obj))

            # make the dirs and start the downloads
            os.makedirs(self.root_dir, exist_ok=True)
            zip_filename = obj + '.zip'
            DownloadGDrive(self.id_drive_subset_categories[obj], os.path.join(self.root_dir,zip_filename), obj_name=obj)
            zipfile.ZipFile(os.path.join(self.root_dir, zip_filename)).extractall(path=os.path.join(self.root_dir))
            os.remove(os.path.join(self.root_dir, zip_filename))

    def stats(self):
        # get the stats to print
        counts = self.class_counts()

        return "%d samples spanning %d classes (avg %d per class)" % \
               (len(self.data), len(counts), int(float(len(self.data))/float(len(counts))))

    def class_counts(self):
        # calculate the number of samples per category
        counts = {}
        for index in range(len(self.samples)):
            sample_data, sample_gt, sample_label, sample_categorie = self.samples[index]
            if sample_label not in counts:
                counts[sample_label] = 1
            else:
                counts[sample_label] += 1

        return counts

    @staticmethod
    def load_data_split(root_dir, split, idxs_videos, num_videos):
        assert split in ['train', 'val', 'test']

        data = []
        categories = []
        labels = []
        gt = []

        label = 0
        key = lambda x: int(os.path.split(x)[1].split('.')[0])
        if isinstance(idxs_videos[0], int):
            if split == 'train':
                idxs_videos = [list(range(end_idx)) for end_idx in idxs_videos]
            elif split == 'test':
                idxs_videos = [list(range(start_idx, end_idx)) for start_idx, end_idx in zip(idxs_videos, num_videos)]

        for i, object in enumerate(sorted(listdir(root_dir))):
            for idx_video in idxs_videos[i]:
                video = object + '-' + str(idx_video+1)
                for frame in sorted(glob.glob(os.path.join(root_dir, object, video, '*.jpg')), key=key):
                # for idx_frame in range(301):
                #     frame = glob.glob(os.path.join(root_dir, object, video, str(idx_frame)+'.jpg'))[0]
                    data.append(frame)
                    labels.append(label)
                    categories.append(object)
                label += 1
                # Read GT
                file = open(os.path.join(root_dir, object, video, 'groundtruth.txt'), 'r')
                data_file = file.readlines()
                reader = csv.reader(data_file)
                for idx, row in enumerate(reader):
                    gt.append(row)

        return data, gt, labels, categories

    @staticmethod
    def load_img(path, crop=False, tuple_crop=None, angle=0):

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(path).convert('RGB')
        original_image = image.copy()
        if crop:
            if angle > 0:
                w_gt = tuple_crop[2] - tuple_crop[0]
                h_gt = tuple_crop[3] - tuple_crop[1]
                x_center = int(tuple_crop[0] + w_gt/2)
                y_center = int(tuple_crop[1] + h_gt/2)
                r = max(w_gt/2.0, h_gt/2.0)
                initial_tuple_crop = (x_center - 1.5*r, y_center - 1.5*r, x_center + 1.5*r, y_center + 1.5*r)

                image = image.crop(initial_tuple_crop)
                image = image.rotate(angle) # angle in degree

                w, h = image.size
                x_center, y_center = int(w/2), int(h/2)
                crop = (x_center - r, y_center - r, x_center + r, y_center + r)
                tuple_crop = crop
            image = image.crop(tuple_crop)
            original_image = original_image.crop(tuple_crop)
        return image, original_image

    @staticmethod
    def degree2rad(angle):
        return np.pi*angle/180.0
        
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    num_videos =  [7, 2, 12, 13, 12, 8]
    idxs_videos = [4, 1, 8, 9, 8, 5]
    split = 'test'
    crop = True
    rotate_image = False
    # root_dir = config.dataset.root_dir
    root_dir = "/home/josev/Documents/dummy_lasot"
    dataset = LaSOTDataset_mod(root_dir,
                              idxs_videos,
                              num_videos,
                              split,
                              crop,
                              transform=None,
                              rotate_image=rotate_image)

    print(dataset.stats())

    plt.figure()
    for i in range(64):
        plt.subplot(8, 8, i+1)
        # idx = np.random.randint(0, dataset.__len__())
        image = dataset.__getitem__(i)[0]
        plt.imshow(image)
        plt.axis('off')
    plt.show()
