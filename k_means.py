import numpy as np
import warnings

class k_means():
    def __init__(self, k, max_it=100):
        self.k = k
        self.max_it = max_it
    
    def fit(self, X):
        # initialize
        self.centroids = self.initialize_clusters(X, k)
        self.initial_centroids = self.centroids.copy()

        # k-means algorithm
        for it in range(self.max_it):
            # assignment step
            clusters = self.predict(X)

            # update step
            centroids = self.compute_centroids(X, clusters, k)

            if np.any(self.centroids != centroids):
                self.centroids = centroids
            else:
                # convergence achieved
                break
        else:
            message = f'Warning: Algorithm didn\'t converge after {self.max_it} iterations.'
            print(message)

        self.total_iterations = it
            

    @staticmethod
    def initialize_clusters(X, k, method='k_means++'):
        
        if method == 'k_means++':
            # select initial point
            idx = [np.random.choice(k)]
            K = np.arange(len(X))

            # initialize weights
            prob = np.ones(len(X))
            prob[idx] = 0
            
            # select remainig points
            for _ in range(k-1):
                # distance to last selected point
                d = np.sum((X - X[idx[-1]])**2, axis=1)
                
                # update weights
                prob = prob * d ** 2
                prob = prob / prob.sum()
                
                # select next point
                idx.append(np.random.choice(K, p=prob))

        elif method == 'basic':
            # select k random observations
            idx = np.random.choice(len(X), size=k, replace=False)

        return X[idx]

    def predict(self, X):
        return np.array([np.sum((self.centroids - x)**2, axis=1).argmin() for x in X])

    @staticmethod
    def compute_centroids(X, clusters, k):
        centroids = []
        for cluster in range(k):
            centroid = X[clusters==cluster].mean(axis=0)
            centroids.append(centroid)

        return np.vstack(centroids)

        


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # params
    n = 100
    k = 5
    X, y = make_blobs(n_samples=n, n_features=2, centers=k, random_state=69)
    print(np.unique(y))
    # fit data
    model = k_means(k)
    model.fit(X)

    # predicted clusters
    y_pred = model.predict(X)

    # plot
    fig, ax = plt.subplots()
    for cluster in range(k):
        f = y_pred == cluster
        ax.plot(X[f, 0], X[f, 1], 'o', label=cluster)
    
    ax.plot(model.initial_centroids[:, 0], model.initial_centroids[:, 1], 'xr', ms=10, label='initial centroids')
    ax.plot(model.centroids[:, 0], model.centroids[:, 1], 'ok', ms=10, label='final centroids')
    # ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_title(f'K-means: {model.total_iterations} iterations')

    plt.show()
 
   












