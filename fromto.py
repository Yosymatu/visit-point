#%%
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
#%%
data = pd.read_csv(r"E:\01_LAB\2023_Lab\10_Data\230814_スケジュール結果_lrs.csv")
data = data.sort_values(["dailyid", "start_time"]).astype({'tatemon_id': 'int32'})
data[data.duration>=15]
# %%
df = data.values

#%%
# #%%
# grouped = gdf_within.groupby('dailyid')

# # 結果を格納する空のリスト
# results = []
# #%%
# # personごとにFromToの表を作成
# for name, group in grouped:
#     # groupを転置
#     group_t = group.T
#     # 列名をリセット
#     group_t = group_t.reset_index()
#     # 列名の変更
#     group_t.columns = ['To', 'From'] + list(group_t.columns[2:])
#     # 'From'列のデータをシフトして'From'列を作成
#     group_t['From'] = group_t['To'].shift(-1)
#     # 最後の行の'From'列を削除
#     group_t = group_t.iloc[:-1]
#     # 'From'列の空白を削除
#     group_t = group_t.dropna()
#     # インデックスをリセット
#     group_t = group_t.reset_index(drop=True)
#     # person名を追加
#     group_t['dailyid'] = name
#     # 結果をリストに追加
#     results.append(group_t)

# %%
data
# %%
# personごとにFromToの出現回数をカウント
unique_persons = np.unique(df[:, 1])

# %%
results = []
for person in unique_persons:
    person_data = df[df[:, 1] == person, 3]
    from_to, counts = np.unique([(person_data[i], person_data[i+1]) for i in range(len(person_data)-1)], axis=0, return_counts=True)
    if len(from_to) == 0:
        pass
    else:
        from_to_counts = np.column_stack((from_to, counts))
        person_column = np.full((from_to_counts.shape[0], 1), person)
        from_to_counts = np.column_stack((person_column, from_to_counts))
        results.append(from_to_counts)
# %%
# FromToの表を結合
results_array = np.vstack(results)
# %%
# personごとのFromToの表を集計
from_to_counts, counts = np.unique(results_array[:, 1:3], axis=0, return_counts=True)
from_to_counts_with_counts = np.column_stack((from_to_counts, counts.reshape(-1, 1)))

# 結果を表示
print(from_to_counts_with_counts)
# %%
df = pd.DataFrame(from_to_counts_with_counts, columns=("To", "From", "Count"))
df.to_csv(r"E:\01_LAB\2023_Lab\10_Data\230814_from_to_counts_with_counts_lrs.csv", encoding="shiftjis")
# %%
gpoi_gdf = gpd.read_file(r"E:\50_DATABASE\google_poi\gpoi_range_lrs.shp")
gpoi_gdf = gpoi_gdf[["tatemon_id", "geometry", "type"]]
centroids = gpoi_gdf.centroid
gpoi_gdf = gpoi_gdf.assign(centroid=centroids).drop(["geometry"],axis=1)
# %%
df = df.astype({"To":"Int64", "From":"Int64", "Count":"Int64"})
# %%
df = pd.merge(df, gpoi_gdf, left_on="To", right_on="tatemon_id", how="left")
df = pd.merge(df, gpoi_gdf, left_on="From", right_on="tatemon_id", how="left")
# %%
def create_line(row):
    return LineString([row["centroid_x"], row["centroid_y"]])
# %%
df["geometry"] = df.apply(create_line, axis=1)



# %%
lines = gpd.GeoDataFrame(df[["To", "From", "Count"]], geometry=df.geometry)
# %%
lines.to_file(r"E:\50_DATABASE\from_to\230814_fromto_lrs.shp")
# %%
df.to_csv(r"E:\01_LAB\2023_Lab\10_Data\230814_FromTo_lrs.csv")
# %%
