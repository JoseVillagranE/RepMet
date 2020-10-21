"""
EpisodeBatchSampler: yield a batch of indexes at each iteration.

Modified from orobix's PrototypicalBatchSampler:
https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/
"""
import numpy as np
import torch


class EpisodeBatchSampler(object):

    def __init__(self, labels, categories_per_epi, num_samples, episodes, perm=True,
                video_labels=None):
        """
        :param labels: iterable containing all the labels for the current dataset (non-uniqued)
        :param categories_per_epi: number of random categories for each episode
        :param num_samples: number of samples for each episode for each class
        :param episodes: number of episodes (iterations) per epoch
        """

        super(EpisodeBatchSampler, self).__init__()

        # set instance variables
        self.labels = labels
        self.categories_per_epi = categories_per_epi
        self.sample_per_class = num_samples
        self.episodes = episodes
        self.perm = perm
        self.video_labels = video_labels

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)
        self.video_classes = []
        if self.video_labels:
            self.video_classes, self.counts = np.unique(self.video_labels, return_counts=True)

        # create a matrix of sample indexes with dim: (num_classes, max(numel_per_lass))
        # fill it with nans
        self.idxs = range(len(self.labels))
        if self.video_labels:
            self.indexes = np.empty((len(self.classes), len(self.video_classes), max(self.counts)), dtype=int) * np.nan
        else:
            self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)

        if self.video_labels:
            self.numel_per_class = torch.zeros((len(self.classes), len(self.video_classes)))
            for idx, (label, video_label) in enumerate(zip(self.labels, self.video_labels)):
                label_idx = np.argwhere(self.classes == label).item()
                video_idx = np.argwhere(self.video_classes == video_label).item()
                self.indexes[label_idx, video_idx, np.where(np.isnan(self.indexes[label_idx, video_idx]))[0][0]] = idx
                self.numel_per_class[label_idx, video_idx] += 1
        else:
            self.numel_per_class = torch.zeros_like(self.classes)
            for idx, label in enumerate(self.labels):
                # for every class c, fill the relative row with the indices samples belonging to c
                label_idx = np.argwhere(self.classes == label).item()
                self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx

                # in numel_per_class we store the number of samples for each class/row
                self.numel_per_class[label_idx] += 1
    def __iter__(self):
        """
        yield a batch of indexes
        """
        spc = self.sample_per_class
        cpi = self.categories_per_epi
        for it in range(self.episodes):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            if self.perm:
                c_idxs = torch.randperm(len(self.classes))[:cpi]
            else:
                c_idxs = torch.arange(len(self.classes))[:cpi]

            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                if self.video_labels:
                    sample_idxs = [torch.randperm(self.numel_per_class[label_idx, j].int().item())[0].item() for j in range(len(self.video_classes))]
                    sample_idxs = torch.from_numpy(np.array(sample_idxs)).long()
                    batch[s] = self.indexes[label_idx, torch.arange(len(self.video_classes)), sample_idxs]
                else:
                    sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                    batch[s] = self.indexes[label_idx][sample_idxs]

            if self.perm:
                batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        return self.episodes


if __name__ == "__main__":
    # use this for debugging and checks
    from utils.debug import set_working_dir
    from config.config import config
    from data_loading.sets.LaSOT import LaSOTDataset
    from torchvision import transforms as trns

    # set the working directory as appropriate
    set_working_dir()

    # load the dataset
    target_obj = ["coin", "cup", "rubicCube"]
    transforms = trns.Compose([trns.Resize((240, 240)),
                               trns.ToTensor()])

    dataset = LaSOTDataset(root_dir=config.dataset.root_dir, transform=transforms, rate_sample=30,
                            categories_subset=target_obj, drive=True, split='test')

    # setup the the sampler
    sampler = EpisodeBatchSampler(labels=dataset.labels, categories_per_epi=3, num_samples=5, episodes=3, perm=False, video_labels=dataset.video_labels)

    # setup the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    for epoch in range(1):
        print("Epoch %d" % epoch)
        for batch in iter(dataloader):
            print('-'*10)
            x, y, z = batch
            print(x)
            print(y)
            print(z)
