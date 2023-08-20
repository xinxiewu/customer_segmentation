"""
models.py contains clustering algorithms:
    1. Centroid-based: K-means
"""
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import silhouette_score

# kmeans()
def kmeans(df_list=None, cluster_low=2, cluster_high=None, fname='km_res.png',
           output=r'../public/output', subplots=[3,2], figsize=(20,30)):
    """ Generate plots of k-means results and save

    Args:
        df_list: list of input dataframe
        
    Returns:
        fname
    """
    sns.set()
    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)
    ax, k, x = axes.flatten(), 0, [w for w in range(cluster_low,cluster_high+1)]
    for j, df in enumerate(df_list):
        ssw, score = [], []
        for i in range(cluster_low,cluster_high+1):
            km = KMeans(n_clusters=i, n_init= 'auto')
            km.fit_predict(df)
            score.append(silhouette_score(df, km.labels_, metric='euclidean'))
            ssw.append(km.inertia_)
        temp = sns.lineplot(x=x, y=ssw, ax=ax[k])
        temp.set(xlabel=f"# of Clusters", ylabel='SSW', title=f"Elbow Method of df {j}")
        k += 1
        temp = sns.lineplot(x=x, y=score, marker='o', ax=ax[k])
        temp.set(xlabel=f"# of Clusters", ylabel='Silhouette Score', title=f"Silhouette Analysis of df {j}")
        k += 1
    plt.tight_layout()
    plt.savefig(os.path.join(output, fname))
    return fname
