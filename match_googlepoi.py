# %%
import pandas as pd
import geopandas as gpd
# %%
df = pd.read_csv(r"E:\50_DATABASE\google_poi\google_poi.csv")
dftype = pd.read_csv(r"E:\50_DATABASE\google_poi\google_poi_type.csv")
# %%
df=df[["lat", "lng", "name", "type0", "type1", "type2"]]
# %%
target = dftype[dftype.target==1]
df["target"] = 0
df["type"] = None


# %%
for key in target.typename.values:
    print(key)
    df["target"] = df.apply(lambda row: 1 if row.type0 == key else row.target, axis = 1)
    df["target"] = df.apply(lambda row: 1 if row.type1 == key else row.target, axis = 1)
    df["target"] = df.apply(lambda row: 1 if row.type2 == key else row.target, axis = 1)
    df["type"] = df.apply(lambda row: row.type2 if row.type2 == key else row.type, axis = 1)
    df["type"] = df.apply(lambda row: row.type1 if row.type1 == key else row.type, axis = 1)
    df["type"] = df.apply(lambda row: row.type0 if row.type0 == key else row.type, axis = 1)

# %%
# %%
df_clenan= df.groupby(["lat", "lng", "name", "type"]).max().reset_index()
# %%
target_df = df_clenan[df_clenan.target == 1][["lat", "lng", "name", "type", "target"]]
target_gdf = gpd.GeoDataFrame(target_df, geometry=gpd.points_from_xy(target_df.lng, target_df.lat), crs=4326)

# %%
gdf_tatemono = gpd.read_file(r"E:\50_DATABASE\zenrin_shape\ippantatemono_clean.shp")
gdf_tatemono = gdf_tatemono[["tatemon_id", "geometry"]]
gdf_tatemono = gdf_tatemono.to_crs(4326)
# %%
gdf = gpd.sjoin(target_gdf, gdf_tatemono, how="left", predicate='intersects')

# %%
# gdf.groupby("tatemon_id").agg(lambda x : x.sum() if x.dtype=='float64' else '/'.join(x)).reset_index().drop(["lat", "lng", "index_right"], axis=1)
gdf = gdf.groupby("tatemon_id").agg(lambda x: x.value_counts().idxmax()).reset_index().drop(["lat", "lng", "index_right", "geometry", "target"], axis=1)
# gdf.to_file(r"E:\50_DATABASE\google_poi\gpoi2.shp")
# %%
gdf = pd.merge(gdf_tatemono, gdf, how="left", on="tatemon_id").dropna()
gdf
# %%
gdf.to_file(r"E:\50_DATABASE\google_poi\gpoi.shp")
# %%
gdf.to_csv(r"E:\50_DATABASE\google_poi\tatemon_poi.csv")
# %%
