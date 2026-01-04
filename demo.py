## DEMO: KMeans Practicals
## 12.12.2025

from sklearn.metrics import silhouette_score

# use e.g. algorithm = 'lloyd'', init = 'k-means++', n_init=10 for KMeans
#he uses km.fit instead of predict - difference?
# plots test_wcss, wcss goes down with nr of clusters
# plots silhouette, has a clear maximum
# get wcss from model after fitting by calling model.inertia_



