import geopandas as gpd
from shapely.geometry import Point

class POISearcher:
    def __init__(self, poi_geojson_file, target_crs="EPSG:6676"):
        print("[Init] Loading POI GeoJSON...")
        self.gdf = gpd.read_file(poi_geojson_file)
        self.target_crs = target_crs
        self.gdf_projected = self.gdf.to_crs(target_crs)
        
        self.gdf_projected['centroid_geom'] = self.gdf_projected.geometry.centroid
        centroids_wgs84 = self.gdf_projected['centroid_geom'].to_crs("EPSG:4326")
        self.gdf_projected['center_lat'] = centroids_wgs84.y
        self.gdf_projected['center_lon'] = centroids_wgs84.x

    def search_nearby(self, current_lat, current_lon, radius_m=2000, limit=10):
        current_point = gpd.GeoSeries([Point(current_lon, current_lat)], crs="EPSG:4326").to_crs(self.target_crs).iloc[0]
        distances = self.gdf_projected.geometry.distance(current_point)
        mask = distances <= radius_m
        nearby = self.gdf_projected[mask].copy()
        nearby['dist'] = distances[mask]
        
        candidates = []
        for _, row in nearby.sort_values('dist').iterrows():
            if row['dist'] < 1.0: continue 
            name = row.get('name', 'Unknown')
            cat = row.get('category', 'Unknown')
            if name != 'Unknown' and cat != 'Unknown':
                candidates.append({"name": name, "category": cat, "dist": int(row['dist'])})
            if len(candidates) >= limit: break
        return candidates

    def get_coords_by_name(self, spot_name):
        target = self.gdf_projected[self.gdf_projected['name'] == spot_name]
        if len(target) > 0:
            return target.iloc[0]['center_lat'], target.iloc[0]['center_lon']
        return None