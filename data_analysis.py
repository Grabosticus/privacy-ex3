import pandas as pd

df = pd.read_csv("filtered_impressionist_clusters.csv")

print(df["Cluster_ind"].value_counts())

