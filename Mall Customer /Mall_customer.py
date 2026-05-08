import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Mall_Customers.csv")

print(df.shape)

print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

# selecting columns
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# scale the data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Build KMeans Model
Kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
Kmeans.fit(X_scaled)

# add Cluster labels
df["Cluster"] = Kmeans.labels_

# See Results
print("Cluster counts:")
print(df["Cluster"].value_counts())

print("\nCluster averages:")
print(df.groupby("Cluster")[
    ["Annual Income (k$)",
     "Spending Score (1-100)"]
].mean())