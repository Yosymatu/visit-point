import pandas as pd
import geopandas as gpd

class POIMatcher:
    def __init__(self, stay_file, poi_geojson_file, output_file, dist_threshold_m=30, target_crs="EPSG:6676"):
        self.stay_file = stay_file
        self.poi_geojson_file = poi_geojson_file
        self.output_file = output_file
        self.dist_threshold_m = dist_threshold_m
        self.target_crs = target_crs

    def process(self):
        print("[Phase 2] Matching Stays with Building Polygons...")
        df_stays = pd.read_csv(self.stay_file)
        if len(df_stays) == 0:
            print("No stays found.")
            return

        gdf_stays = gpd.GeoDataFrame(
            df_stays,
            geometry=gpd.points_from_xy(df_stays['longitude'], df_stays['latitude']),
            crs="EPSG:4326"
        ).to_crs(self.target_crs)
        
        gdf_poi = gpd.read_file(self.poi_geojson_file).to_crs(self.target_crs)
        
        joined = gpd.sjoin_nearest(
            gdf_stays, gdf_poi, 
            how="left", 
            max_distance=self.dist_threshold_m, 
            distance_col="dist_to_poi"
        )
        
        name_col = 'name' if 'name' in joined.columns else 'building_name'
        cat_col = 'category' if 'category' in joined.columns else 'type'
        
        joined['poi_name'] = joined[name_col].fillna('Unknown')
        joined['poi_category'] = joined[cat_col].fillna('Unknown')
        
        output_cols = ['uuid', 'stay_start_time', 'stay_end_time', 'duration_min', 'latitude', 'longitude', 'poi_name', 'poi_category']
        joined[output_cols].drop_duplicates(subset=['uuid', 'stay_start_time']).to_csv(self.output_file, index=False)
        print(f"Saved matched trip chains to {self.output_file}")