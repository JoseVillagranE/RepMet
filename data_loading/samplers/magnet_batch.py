"""
EpisodeBatchSampler: yield a batch of indexes at each iteration.

Modified from
"""
import numpy as np
import torch
from sklearn.cluster import KMeans


class MagnetBatchSampler(object):

    def __init__(self, labels, k, m, d, iterations):
        """
        :param labels: iterable containing all the labels for the current dataset (non-uniqued)
        :param k: the number of clusters per class
        :param m: the number of clusters to sample
        :param d: the number of examples to sample per cluster
        """

        super(MagnetBatchSampler, self).__init__()

        # set instance variables
        self.labels = np.array(labels)
        self.k = k
        self.m = m
        self.d = d
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)
        self.num_classes = len(self.classes)

        # make sure we have enough clusters to chose m from
        assert self.m < (self.num_classes*self.k), "Not enough classes in this dataset for m: %d" % self.m

        self.centroids = None
        self.assignments = np.zeros_like(labels, int)

        self.cluster_assignments = {}
        self.cluster_classes = np.repeat(range(self.num_classes), k)
        self.example_losses = None
        self.cluster_losses = None
        self.has_loss = None

        self.batch_indexes = None

    def __iter__(self):
        """
        yield a batch of indexes
        """
        for it in range(self.iterations):
            batch, _ = self.gen_batch()
            yield batch

    def __len__(self):
        return self.iterations

    def update_clusters(self, rep_data, max_iter=20):
        """Given an array of representations for the entire training set,
        recompute clusters and store example cluster assignments in a
        quickly sampleable form."""
        # Lazily allocate array for centroids
        if self.centroids is None:
            self.centroids = np.zeros([self.num_classes * self.k, rep_data.shape[1]])

        for c in range(self.num_classes):
            class_mask = self.labels == c
            class_examples = rep_data[class_mask]
            kmeans = KMeans(n_clusters=self.k, init='k-means++', n_init=1, max_iter=max_iter)
            kmeans.fit(class_examples)

            # Save cluster centroids for finding impostor clusters
            start = self.get_cluster_ind(c, 0)
            stop = self.get_cluster_ind(c, self.k)
            self.centroids[start:stop] = kmeans.cluster_centers_

            # Update assignments with new global cluster indexes
            self.assignments[class_mask] = self.get_cluster_ind(c, kmeans.predict(class_examples))

        # Construct a map from cluster to example indexes for fast batch creation
        for cluster in range(self.k * self.num_classes):
            cluster_mask = self.assignments == cluster
            self.cluster_assignments[cluster] = np.flatnonzero(cluster_mask)

    def update_losses(self, losses):
        """Given a list of examples indexes and corresponding losses
        store the new losses and update corresponding cluster losses."""
        # Lazily allocate structures for losses
        if self.example_losses is None:
            self.example_losses = np.zeros_like(self.labels, float)
            self.cluster_losses = np.zeros([self.k * self.num_classes], float)
            self.has_loss = np.zeros_like(self.labels, bool)

        losses = losses.data.cpu().numpy()

        self.example_losses[self.batch_indexes] = losses
        self.has_loss[self.batch_indexes] = losses

        epsilon = 1e-8

        # Find affected clusters and update the corresponding cluster losses
        clusters = np.unique(self.assignments[self.batch_indexes])
        for cluster in clusters:
            cluster_inds = self.assignments == cluster
            cluster_example_losses = self.example_losses[cluster_inds]

            # Take the average closs in the cluster of examples for which we have measured a loss
            inds = cluster_example_losses[self.has_loss[cluster_inds]]

            if len(inds) < 1:
                self.cluster_losses[cluster] = epsilon
            else:
                self.cluster_losses[cluster] = np.mean(cluster_example_losses[self.has_loss[cluster_inds]]) + epsilon

    def gen_batch(self):
        """Sample a batch by first sampling a seed cluster proportionally to
        the mean loss of the clusters, then finding nearest neighbor
        "impostor" clusters, then sampling d examples uniformly from each cluster.

        The generated batch will consist of m clusters each with d consecutive
        examples."""

        # Sample seed cluster proportionally to cluster losses if available
        if self.cluster_losses is not None:
            p = self.cluster_losses / np.sum(self.cluster_losses)
            seed_cluster = np.random.choice(self.num_classes * self.k, p=p)
        else:
            seed_cluster = np.random.choice(self.num_classes * self.k)

        # Get imposter clusters by ranking centroids by distance
        sq_dists = ((self.centroids[seed_cluster] - self.centroids) ** 2).sum(axis=1)

        # Assure only clusters of different class from seed are chosen
        sq_dists[self.get_class_ind(seed_cluster) == self.cluster_classes] = np.inf

        # Get top impostor clusters and add seed
        clusters = np.argpartition(sq_dists, self.m - 1)[:self.m - 1]
        clusters = np.concatenate([[seed_cluster], clusters])

        # print(clusters)  # debug print the clusters per batch

        # Sample examples uniformly from cluster
        batch_indexes = np.empty([self.m * self.d], int)
        for i, c in enumerate(clusters):
            # TODO sometimes arent enough cluster assignments available, what do we do then?
            # if you get this you can set k = 1 (one cluster per class)
            x = np.random.choice(self.cluster_assignments[c], self.d, replace=False) # replace=False)  # sometimes there ain't enough d in assignments...
            start = i * self.d
            stop = start + self.d
            batch_indexes[start:stop] = x

        # Translate class indexes to index for classes within the batch
        class_inds = np.array([self.get_class_ind(c) for c in clusters])
        batch_class_inds = []
        inds_map = {}
        class_count = 0
        for c in class_inds:
            if c not in inds_map:
                inds_map[c] = class_count
                class_count += 1
            batch_class_inds.append(inds_map[c])

        self.batch_indexes = batch_indexes  # this is used to update losses

        return batch_indexes, np.repeat(batch_class_inds, self.d)

    def get_cluster_ind(self, c, i):
        """Given a class index and a cluster index within the class
        return the global cluster index"""
        return c * self.k + i

    def get_class_ind(self, c):
        """Given a cluster index return the class index."""
        return int(c / self.k)  # int floors it for us nicely, without this you could get a floating value


# if __name__ == "__main__":
#     # use this for debugging and checks
#     from utils.debug import set_working_dir
#     from config.config import config
#     from data_loading.sets import OxfordFlowersDataset, OmniglotDataset
#
#     # set the working directory as appropriate
#     set_working_dir()
#
#     # load the dataset
#     dataset = OxfordFlowersDataset(root_dir=config.dataset.root_dir)
#
#     # setup the the sampler
#     sampler = MagnetBatchSampler(labels=dataset.labels, k=8, m=12, d=4, iterations=5)
#
#     # setup the dataloader
#     dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
#
#
#     # dataloader.sampler.update_clusters(reps)
#     for epoch in range(1):
#         print("Epoch %d" % epoch)
#         for batch in iter(dataloader):
#             print('-'*10)
#             x, y = batch
#             print(x)
#             print(y)
