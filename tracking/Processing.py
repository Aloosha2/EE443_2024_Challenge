from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

class postprocess:
    
    def __init__(self, number_of_people, cluster_method):
        """
        Initializes the postprocess class with the number of clusters and the chosen clustering method.
        
        Parameters:
        -----------
        number_of_people : int
            The number of clusters (people) to form.
        cluster_method : str
            The clustering method to use. Supported values are 'kmeans', 'agglomerative', and 'dbscan'.
        
        Raises:
        -------
        NotImplementedError
            If an unsupported clustering method is provided.
        """
        self.n = number_of_people
        if cluster_method == 'kmeans':
            self.cluster_method = KMeans(n_clusters=self.n, random_state=0)
        elif cluster_method == 'agglomerative':
            self.cluster_method = AgglomerativeClustering(n_clusters=self.n)
        elif cluster_method == 'dbscan':
            self.cluster_method = DBSCAN()
        else:
            raise NotImplementedError(f"Clustering method {cluster_method} is not implemented")
    
    def run(self, features):
        """
        Runs the clustering algorithm on the provided features and returns the cluster labels.
        
        Parameters:
        -----------
        features : array-like, shape (n_samples, n_features)
            The feature data to cluster.
        
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels for each point in the dataset.
        """
        print('Start Clustering')
        
        # Fit the clustering method to the provided features
        self.cluster_method.fit(features)
        
        print('Finish Clustering')

        # Return the labels (cluster assignments) if they are available
        if hasattr(self.cluster_method, 'labels_'):
            return self.cluster_method.labels_
        else:
            # If not, use the predict method to get cluster assignments
            return self.cluster_method.predict(features)
