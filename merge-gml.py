# %%
import geopandas as gpd
import os
import osmnx as ox
import pandas as pd
# %%
place = "Nakamuraku,Nagoya,Aichi,Japan"
tags = {'building': True}
gdf = ox.geometries_from_place(place, tags)
# %%
bldls = ["commercial", "retail", "supermarket"]
ls = []
for i in bldls:
    gdf_tmp = gdf[gdf.building == i]
    ls.append(gdf_tmp)

# %%
place = "Nakaku,Nagoya,Aichi,Japan"
gdf = ox.geometries_from_place(place, tags)
# %%
for i in bldls:
    gdf_tmp = gdf[gdf.building == i]
    ls.append(gdf_tmp)
# %%
merged_gdf = pd.concat(ls, ignore_index=True)
# %%
merged_gdf.to_file("osm_commercial.geojson", driver='GeoJSON')
# %%
merged_gdf["geometry"].to_file("osm_commercial.geojson", driver='GeoJSON')
# %%
