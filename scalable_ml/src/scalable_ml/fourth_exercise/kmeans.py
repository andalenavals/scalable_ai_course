"""
Module that implements Kmeans algorithm
"""
import torch
from torchmetrics.functional.pairwise import pairwise_euclidean_distance


class KMeans:
    r"""
    K-Means clustering algorithm. An implementation of Lloyd's algorithm [1].

    Attributes
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float
        Relative tolerance with regard to inertia to declare convergence.
    repetition_nr : int
        Determines the number of repetitions of KMeans algorithms (solution with lowest inertia is returned)

    Notes
    -----
    The average complexity is given by :math:`O(k \cdot n \cdot T)`, were n is the number of samples and :math:`T` is the number of iterations.
    In practice, the k-means algorithm is very fast, but it may fall into local minima. That is why it can be useful
    to restart it several times.

    References
    ----------
    [1] Lloyd, Stuart P., "Least squares quantization in PCM", IEEE Transactions on Information Theory, 28 (2), pp.
    129–137, 1982.
    """

    def __init__(self, n_clusters=2, max_iter=10, tol=1e-12, repetition_nr=6):
        self._cluster_centers = None
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol
        self._repetition_nr = repetition_nr

    def _initialize_cluster_centers(self, x):

        # In the beginning, we just perform a random assigment of the data to the clusters
        local_labels_init = torch.randint(0, self._n_clusters, (x.shape[0],))
        # Your Code ...


    def _assign_to_cluster(self, x):
        """
        Assigns each observation to the cluster with the nearest mean

        Parameters
        ----------
        x : pytorch tensor
            Data points, Shape = (n_samples, n_features)
        """
        # calculate the distance matrix and determine the closest centroid
        distances = pairwise_euclidean_distance(x, self._cluster_centers).pow(2)
        matching_centroids = distances.argmin(axis=1)

        return matching_centroids

    def _update_centroids(self, x, local_labels):
        """
        Compute coordinates of new centroid as mean of the data points in ``x`` that are assigned to this centroid.

        Parameters
        ----------
        x :  pytorch tensor
            Input data
        local_labels : pytorch tensor
            Array filled with indices ``i`` indicating to which cluster ``ci`` each sample point in ``x`` is assigned

        """
        new_cluster_centers = torch.empty_like(self._cluster_centers)

        # Your Code ...

        # points in current cluster
        # Your Code ...

        # accumulate points and total number of points in cluster
        # Your Code ...

        # compute the new centroids
        # Your Code ...

        return new_cluster_centers

    def fit(self, x):
        """
        Computes the centroid of a k-means clustering.

        Parameters
        ----------
        x : pytorch tensor
            Training instances to cluster. Shape = (n_samples, n_features)

        """

        # Cluster center for all repetitions
        cluster_candidates = torch.zeros(self._repetition_nr, self._n_clusters, x.shape[1])
        labels_candidates = torch.zeros(self._repetition_nr, x.shape[0])

        for rep in range(self._repetition_nr):
            print(f'Repetition: {rep}')

            # initialize the clustering
            self._initialize_cluster_centers(x)
            inertia = 1e6
            n_iter = 0

            # iteratively fit the points to the centroids
            for epoch in range(self._max_iter):
                # increment the iteration count
                n_iter += 1
                print(f'Iteration: {n_iter}  Inertia: {inertia}')

                # assignment step
                local_labels = self._assign_to_cluster(x)

                # update step
                new_cluster_centers = self._update_centroids(x, local_labels)
                # check whether centroid movement has converged
                # Your Code ...


        # return solution with best (i.e. smallest) inertia
        # Your Code ...
        best_run = 42 # ToDO: find best run
        
        return labels_candidates[best_run, :], cluster_candidates[best_run, :, :] 
