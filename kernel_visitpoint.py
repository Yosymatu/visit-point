import pandas as pd
import geopandas as gpd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
from shapely.geometry import Polygon, MultiPolygon
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy import MetaData



def calculate_kde(sample_num, gdf):
    x = gdf.longitude
    y = gdf.latitude
    wg = gdf.accuracy

    xy = np.vstack([x,y])
    try:
        return gaussian_kde(xy, bw_method='silverman',weights=wg)
        # return gaussian_kde(xy, bw_method='silverman')
    except:
        return None

def calculate_50pcts_con(kernel, gdf):
    xmin = gdf.longitude.min()
    xmax = gdf.longitude.max()
    ymin = gdf.latitude.min()
    ymax = gdf.latitude.max()

    X, Y = np.mgrid[xmin:xmax:2000j, ymin:ymax:2000j]
    levels = [0.5,1] #0.5が50％カーネル密度

    cfset = plt.contourf(X, Y, np.reshape(kernel(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)/np.max(np.reshape(kernel(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)), levels = levels, cmap='Blues')
    plt.close()
    return cfset

def center_to_sql(sample_num, cfset):
    from sqlalchemy import insert
    for col in cfset.collections:
        # Loop through all polygons that have the same intensity level
        for contour in col.get_paths():
            # Create a polygon for the countour
            # First polygon is the main countour, the rest are holes
            for ncp,cp in enumerate(contour.to_polygons()):
                x = cp[:,0]
                y = cp[:,1]
                new_shape = Polygon([(i[0], i[1]) for i in zip(x,y)]).centroid
                if ncp == 0:
                    poly = new_shape
                else:
                    # Remove holes, if any
                    poly = poly.difference(new_shape)
                # Append polygon to list
                # print('===== INSERT')
                stmt = (
                    insert(PDP_0006_20211106_SP).
                    values(dailyid=sample_num, longitude=poly.x, latitude=poly.y)
                )
                engine.execute(stmt)

def main():
    engine = create_engine("postgresql://postgres:password@localhost/agoop_db")
    dbmetadata = MetaData()
    dbmetadata.reflect(bind=engine)
    PDP_0006_20211106_SP = dbmetadata.tables['pdp_0006_20211106_sp']

    dailyid_unique_ls = pd.read_csv("PDP_0006_20211106_dailyid_unique.csv")
    n = 0
    for sample in dailyid_unique_ls["0"][7340:]:
        print(sample)
        gdf = pd.read_sql("SELECT * FROM PDP_0006_20211106 WHERE dailyid = '{0}' AND month = 11 AND day = 6 AND os = 'Android' AND accuracy <= 20 AND speed < 3.0".format(sample), engine)
        gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf.longitude, gdf.latitude), crs="EPSG:4612")

        kernel_res = calculate_kde(sample, gdf)
        if kernel_res == None:
            pass
        else:
            cfset = calculate_50pcts_con(kernel_res, gdf)
            center_to_sql(sample, cfset)
            del gdf
            del kernel_res
            del cfset
