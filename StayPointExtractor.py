import pandas as pd
import geopandas as gpd

class POIMatcher:
    """
    滞在地点データ(CSV)とPOIデータ(GeoJSON/Polygon)を空間結合するクラス。
    """

    def __init__(self, stay_file, poi_geojson_file, output_file, 
                 dist_threshold_m=50, 
                 target_crs="EPSG:6676"):
        """
        Args:
            stay_file (str): 滞在地点CSVのパス
            poi_geojson_file (str): POIのGeoJSONファイル(.geojson)
            output_file (str): 出力ファイルのパス
            dist_threshold_m (int): 許容距離(m)。建物ポリゴンからの距離。
            target_crs (str): 距離計算用の投影座標系(長野はEPSG:6676)
        """
        self.stay_file = stay_file
        self.poi_geojson_file = poi_geojson_file
        self.output_file = output_file
        self.dist_threshold_m = dist_threshold_m
        self.target_crs = target_crs

    def process(self):
        print("データを読み込んでいます...")
        
        # 1. 滞在データの読み込み & GeoDataFrame化
        df_stays = pd.read_csv(self.stay_file)
        gdf_stays = gpd.GeoDataFrame(
            df_stays,
            geometry=gpd.points_from_xy(df_stays['longitude'], df_stays['latitude']),
            crs="EPSG:4326"
        ).to_crs(self.target_crs)
        
        # 2. POI(GeoJSON)の読み込み
        # GeoJSONは通常WGS84(EPSG:4326)です
        print(f"GeoJSONを読み込み中: {self.poi_geojson_file}")
        gdf_poi = gpd.read_file(self.poi_geojson_file)
        
        # 投影座標系へ変換 (メートル計算のため)
        gdf_poi = gdf_poi.to_crs(self.target_crs)
        
        # 3. 空間結合 (Nearest Neighbor)
        # Point(滞在) と Polygon(建物) の距離を計算します
        print("空間結合(Nearest Neighbor Search)を実行中...")
        
        joined = gpd.sjoin_nearest(
            gdf_stays,
            gdf_poi,
            how="left",
            max_distance=self.dist_threshold_m,
            distance_col="dist_to_poi"
        )
        
        # 4. データ整理
        # GeoJSONのプロパティに合わせて列名を調整してください
        # 例: GeoJSONのプロパティが 'name', 'type' の場合
        poi_name_col = 'name' if 'name' in joined.columns else 'building_name' # 適宜変更
        poi_cat_col = 'category' if 'category' in joined.columns else 'type'   # 適宜変更
        
        joined['poi_name'] = joined[poi_name_col].fillna('Unknown')
        joined['poi_category'] = joined[poi_cat_col].fillna('Unknown')
        
        # 出力カラムの整理
        output_cols = [
            'uuid', 'stay_start_time', 'stay_end_time', 'duration_min',
            'latitude', 'longitude',  # 滞在地点
            'poi_name', 'poi_category', 'dist_to_poi'
        ]
        
        # 重複排除
        final_df = joined[output_cols].drop_duplicates(subset=['uuid', 'stay_start_time'])
        
        # 保存
        print(f"保存中...: {self.output_file}")
        final_df.to_csv(self.output_file, index=False)
        
        match_rate = (final_df['poi_name'] != 'Unknown').mean() * 100
        print(f"処理完了. POIマッチング率: {match_rate:.1f}%")

if __name__ == "__main__":
    matcher = POIMatcher(
        stay_file='./output_stays.csv',
        poi_geojson_file='./ina_buildings.geojson', # GeoJSONを指定
        output_file='./final_trip_chain_poly.csv',
        dist_threshold_m=30, # ポリゴンなら距離閾値はもっと厳しくても良いかもしれません
        target_crs="EPSG:6676"
    )
    matcher.process()
