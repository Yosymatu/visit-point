#%%
import pandas as pd
import geopandas as gpd
from collections import defaultdict
import csv
from csv import reader
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import warnings
np.warnings = warnings
# %%

df = pd.read_csv(r"E:\01_LAB\2023_Lab\10_Data\230814_スケジュール結果.csv")
# %%
gdf = gpd.GeoDataFrame(df.drop("geometry", axis =1), geometry=df.geometry, crs=4326)
# %%
