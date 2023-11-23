# %%
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import warnings
np.warnings = warnings
# %%
df = pd.read_csv((r"E:\01_LAB\2023_Lab\10_Data\230814_スケジュール結果.csv"))
df['duration'] = df['duration'].mask(df['duration']<15, 0)
# %%
df_pivot = pd.pivot_table(df, index="dailyid", columns="google_poi", values = "duration", aggfunc="sum").reset_index()
df_pivot2 = pd.pivot_table(df, index="dailyid", columns="google_poi", values = "type", aggfunc="count").reset_index()
# %%
df_pivot = df_pivot.fillna(0)
df_pivot2 = df_pivot2.fillna(0)
# df_pivot = df_pivot.replace(0, 1)
# %%
df_res = pd.merge(df_pivot, df_pivot2, on = "dailyid", suffixes = ["_sum_duration", "_type_count"])
# %%

X=df_res[df_res.columns.to_list()[1:]].values
X.shape
# %%
# cust_array = np.array([df_pivot['culture'].tolist(),
#                        df_pivot['commerce'].tolist(),
#                        df_pivot['lrs'].tolist(),
#                        df_pivot['tour'].tolist()
#                        ], np.float32)

# %%
sum_of_squared_errors = []
for i in range(1, 20):
    model = KMeans(n_clusters=i, random_state=0, init='random')
    model.fit(X)
    sum_of_squared_errors.append(model.inertia_)  # 損失関数の値を保存

plt.plot(range(1, 20), sum_of_squared_errors, marker='o')
plt.xlabel('number of clusters')
plt.ylabel('sum of squared errors')
plt.show()
# %%
pd.DataFrame(sum_of_squared_errors).to_csv(r"E:\01_LAB\2023_Lab\10_Data\elbow.csv")
# %%
pred = KMeans(n_clusters=8).fit_predict(X)
# %%
df_res['cluster_id']=pred
# %%
df_res['cluster_id'].value_counts()
# %%
df_res
# %%
df_res.to_csv(r"E:\01_LAB\2023_Lab\10_Data\230819_cluster_res3.csv")
# %%
df_cl1 = df_res[df_res.cluster_id == 1]
sum_of_squared_errors = []
for i in range(1, 20):
    model = KMeans(n_clusters=i, random_state=0, init='random')
    model.fit(df_cl1.drop(["cluster_id", "dailyid"], axis=1))
    sum_of_squared_errors.append(model.inertia_)  # 損失関数の値を保存

plt.plot(range(1, 20), sum_of_squared_errors, marker='o')
plt.xlabel('number of clusters')
plt.ylabel('sum of squared errors')
plt.show()
# %%
df_cl1
# %%
pred = KMeans(n_clusters=6).fit_predict(df_cl1.drop(["cluster_id", "dailyid"]))

# %%
df_cl1['cluster_id']=pred
df_cl1['cluster_id'].value_counts()
df_cl1.to_csv(r"E:\01_LAB\2023_Lab\10_Data\230819_cluster_res_cl1.csv")
# %%
####分析２
df_pivot_count = pd.pivot_table(df, index="dailyid", columns="google_poi", values = "duration", aggfunc="count").reset_index()
# %%
df_pivot_count = df_pivot_count.fillna(0)
# %%
df = pd.merge(df_pivot_mean, df_pivot_count, how="left", on="dailyid", suffixes=("","_count"))
# %%
cust_array = np.array([df['culture'].tolist(),
                       df['eat'].tolist(),
                       df['goods'].tolist(),
                       df['lrs'].tolist(),
                       df['tour'].tolist()
                       ], np.float32)
# %%
cust_array = cust_array.T
# %%
pred = KMeans(n_clusters=4).fit_predict(cust_array)
# %%
sum_of_squared_errors = []
for i in range(1, 30):
    model = KMeans(n_clusters=i, random_state=0, init='random')
    model.fit(cust_array)
    sum_of_squared_errors.append(model.inertia_)  # 損失関数の値を保存

plt.plot(range(1, 30), sum_of_squared_errors, marker='o')
plt.xlabel('number of clusters')
plt.ylabel('sum of squared errors')
plt.show()
# %%
sum_of_squared_errors
# %%
pred = KMeans(n_clusters=7).fit_predict(cust_array)
# %%
df['cluster_id']=pred
# %%
df['cluster_id'].value_counts()
# %%
clusterinfo = pd.DataFrame()
for i in range(7):
    clusterinfo['cluster' + str(i)] = df[df['cluster_id']==i][df.columns.to_list()[1:]].mean()

my_plot = clusterinfo.T.plot(kind='bar', stacked=True, title="Mean Value of 4 Clusters")
my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=0)

# %%
df.to_csv(r"E:\01_LAB\2023_Lab\10_Data\230814_cluster_res2.csv")
# %%
linkage_result = linkage(df_res.drop("dailyid", axis=1), method='ward', metric='euclidean')
# %%
threshold = 0.7 * np.max(linkage_result[:, 2])
# 階層型クラスタリングの可視化
plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor='w', edgecolor='k')
dendrogram(linkage_result, labels=df_res.index, color_threshold=threshold)
plt.axhline(7, linestyle='--', color='r')
plt.show()

# %%
df_pivot.to_csv(r"E:\01_LAB\2023_Lab\10_Data\230819_cluster_res.csv")
# %%
X_norm
# %%
X
# %%
linkage_result
# %%
df_cl1
# %%
