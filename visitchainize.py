#%%
import pandas as pd
import geopandas as gpd
from collections import defaultdict
import csv
from csv import reader

# %%
df = pd.read_csv(r"E:\01_LAB\2023_Lab\10_Data\android_holiday_range.csv")
df = df[df["accuracy"] <= 100]
# %%
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
gdf["hour"] = gdf["hour"].astype(float)
gdf["minute"] = gdf["minute"].astype(float)
gdf["time"] = gdf["hour"]*60 + gdf["minute"]
gdf = gdf.sort_values("time")
# %%
gpoi_gdf = gpd.read_file(r"E:\50_DATABASE\google_poi\gpoi_range_lrs.shp")
gpoi_gdf = gpoi_gdf.to_crs(4326)

# %%
joined_data = gpd.sjoin(gdf, gpoi_gdf[["tatemon_id", "geometry", "type"]].to_crs("EPSG:4326"), how="left", predicate='intersects')
# %%
# %%
ls = []
for sample_id, sample_group in joined_data.groupby(['dailyid', 'day']):
    sample_group = sample_group.sort_values(['day','time'])
    current_building = None
    
    for i, row in sample_group.iterrows():
        building_id = row['tatemon_id']
        timestamp = row['time']
        if pd.isnull(row['tatemon_id']):
            pass
        elif current_building is None:
            current_building = building_id
            start_time = timestamp
        elif current_building != building_id:
            duration = timestamp - start_time
            if duration <1:
                pass
            else:
                # stay_duration_per_person[3] += duration
                stay_duration_per_person = [sample_id[0], sample_id[1],current_building, start_time, duration]
                ls.append(stay_duration_per_person)
                
                current_building = building_id
                start_time = timestamp
            
    # 最後の滞在時間を計算する
    if current_building is not None:
        duration = sample_group['time'].iloc[-1] - start_time
        stay_duration_per_person = [sample_id[0], sample_id[1], current_building, start_time, duration]
        ls.append(stay_duration_per_person)
# %%
ls
# %%
# データフレームに変換する
df = pd.DataFrame(ls, columns=['dailyid', "day", 'tatemon_id', 'start_time', 'duration'])
df = df.dropna(subset=['tatemon_id'])

# %%
# df["duration"] = df["duration"].apply(lambda x: x if x >=15 else None)
# %%
df = pd.merge(df, gpoi_gdf, how="left")
# df = df.groupby(['dailyid', "day", 'tatemon_id', 'start_time', 'duration']).max().reset_index()
# %%
gdf = gpd.GeoDataFrame(df.drop("geometry", axis =1), geometry=df.geometry, crs=4326)
# %%
df.to_csv(r"E:\01_LAB\2023_Lab\10_Data\230814_スケジュール結果.csv")
# %%
gdf.geometry = gdf.geometry.centroid
# %%
gdf.to_file(r"E:\01_LAB\2023_Lab\10_Data\230814_schedule_res.shp")
# %%　ここから試行
df = pd.read_csv(r"E:\01_LAB\2023_Lab\10_Data\android_holiday_range.csv")
df = df[df["accuracy"] <= 100]
# %%
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
gdf["hour"] = gdf["hour"].astype(float)
gdf["minute"] = gdf["minute"].astype(float)
gdf["time"] = gdf["hour"]*60 + gdf["minute"]
gdf = gdf.sort_values("time")
# %%
gpoi_gdf = gpd.read_file(r"E:\50_DATABASE\google_poi\gpoi_range_lrs.shp")
gpoi_gdf = gpoi_gdf.to_crs(4326)
gpoi_gdf = gpoi_gdf[gpoi_gdf["type"] == "lrs"]
# %%
joined_data = gpd.sjoin(gdf, gpoi_gdf[["tatemon_id", "geometry", "type"]].to_crs("EPSG:4326"), how="left", predicate='intersects')

# %%
ls = []
for sample_id, sample_group in joined_data.groupby(['dailyid', 'day']):
    sample_group = sample_group.sort_values(['day','time'])
    current_building = None
    
    for i, row in sample_group.iterrows():
        building_id = row['tatemon_id']
        timestamp = row['time']
        if pd.isnull(row['tatemon_id']):
            pass
        elif current_building is None:
            current_building = building_id
            start_time = timestamp
        elif current_building != building_id:
            duration = timestamp - start_time
            if duration <1:
                pass
            else:
                # stay_duration_per_person[3] += duration
                stay_duration_per_person = [sample_id[0], sample_id[1],current_building, start_time, duration]
                ls.append(stay_duration_per_person)
                
                current_building = building_id
                start_time = timestamp
            
    # 最後の滞在時間を計算する
    if current_building is not None:
        duration = sample_group['time'].iloc[-1] - start_time
        stay_duration_per_person = [sample_id[0], sample_id[1], current_building, start_time, duration]
        ls.append(stay_duration_per_person)
# %%
# データフレームに変換する
df = pd.DataFrame(ls, columns=['dailyid', "day", 'tatemon_id', 'start_time', 'duration'])
df = df.dropna(subset=['tatemon_id'])
df = pd.merge(df, gpoi_gdf, how="left")
df.to_csv(r"E:\01_LAB\2023_Lab\10_Data\230814_スケジュール結果_lrs.csv")
# %%
