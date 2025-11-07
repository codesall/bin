import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

# --- load ---
df = pd.read_csv("sales_data_sample.csv", encoding="unicode_escape")

# --- numeric features only (drop obvious non-numeric columns) ---
drop_if_present = ['OrderDate','OrderDateTime','Order ID','Order ID','Order ID','Address','ADDRESSLINE1','ADDRESSLINE2','STATE','POSTALCODE','PHONE','CustomerID']
df = df.drop([c for c in drop_if_present if c in df.columns], axis=1, errors='ignore')
X = df.select_dtypes(include=[np.number]).copy()
X = X.dropna(axis=1, how='all')  # drop empty numeric cols
X = X.fillna(X.median())

# --- scale ---
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# --- elbow: inertia for k=1..10 ---
ks = list(range(1, 11))
inertias = []
for k in ks:
    inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs).inertia_)

# --- automatic elbow detection: distance from line (first-last) ---
# points (k, inertia)
pts = np.column_stack((ks, inertias))
p1, p2 = pts[0], pts[-1]
# line vector
v = p2 - p1
# distances
distances = np.abs(np.cross(v, pts - p1) / np.linalg.norm(v))
optimal_k = int(ks[np.argmax(distances)])
if optimal_k < 2:
    optimal_k = 2

print("Inertia by k:", dict(zip(ks, inertias)))
print("Detected elbow (optimal k):", optimal_k)

# --- final KMeans ---
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(Xs)
df['kmeans_cluster'] = kmeans.labels_

# --- hierarchical clustering (Agglomerative) ---
agg = AgglomerativeClustering(n_clusters=optimal_k).fit(Xs)
df['hier_cluster'] = agg.labels_

# --- outputs ---
print("\nKMeans cluster counts:\n", df['kmeans_cluster'].value_counts().sort_index().to_dict())
print("\nHierarchical cluster counts:\n", df['hier_cluster'].value_counts().sort_index().to_dict())

# compact cluster summary (means of numeric features)
print("\nKMeans cluster centers (in original scale):")
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=X.columns).round(3)
print(centers_df)

# # save labelled data if needed
# df.to_csv("sales_clusters_labeled.csv", index=False)
# print("\nSaved labeled data to sales_clusters_labeled.csv")
