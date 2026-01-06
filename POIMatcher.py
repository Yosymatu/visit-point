import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

class POIMatcher:
    """
    滞在地点データ(CSV)とPOIデータ(CSV)を空間結合するクラス。
    最近傍探索(Nearest Neighbor)を用いて、指定距離内のPOIを付与する。
    """

    def __init__(self, stay_file, poi_file, output_file, 
                 dist_threshold_m=50, 
                 poi_encoding='utf-8',
                 target_crs="EPSG:6676"):
        """
        Args:
            stay_file (str): 滞在地点CSVのパス
            poi_file (str): POIデータCSVのパス
            output_file (str): 出力ファイルのパス
            dist_threshold_m (int): POIとみなす最大距離(メートル)。GPS誤差を考慮し少し広め(30-50m)推奨。
            poi_encoding (str): POIファイルの文字コード(Windowsなら'cp932'の場合あり)
            target_crs (str): 距離計算用の投影座標系。長野県は第8系(EPSG:6676)が最適。
        """
        self.stay_file = stay_file
        self.poi_file = poi_file
        self.output_file = output_file
        self.dist_threshold_m = dist_threshold_m
        self.poi_encoding = poi_encoding
        self.target_crs = target_crs

    def _load_and_convert_to_gdf(self, df, lat_col, lon_col):
        """DataFrameをGeoDataFrameに変換し、投影座標系へ変換する"""
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326"  # 元データはWGS84(緯度経度)と仮定
        )
        return gdf.to_crs(self.target_crs)

    def process(self):
        print("データを読み込んでいます...")
        
        # 1. データの読み込み
        # 滞在データ (前のステップの出力)
        df_stays = pd.read_csv(self.stay_file)
        
        # POIデータ (必要なカラム: name, category, lat, lon など)
        df_poi = pd.read_csv(self.poi_file, encoding=self.poi_encoding)
        
        # POIデータのカラム名チェック（想定される名前に合わせる処理）
        # ※ユーザーのデータに合わせて適宜変更してください
        lat_col_poi = 'latitude' if 'latitude' in df_poi.columns else 'lat'
        lon_col_poi = 'longitude' if 'longitude' in df_poi.columns else 'lon'
        
        # 2. GeoDataFrameへの変換と投影変換(メートル単位へ)
        print(f"座標変換を実行中... (Target CRS: {self.target_crs})")
        gdf_stays = self._load_and_convert_to_gdf(df_stays, 'latitude', 'longitude')
        gdf_poi = self._load_and_convert_to_gdf(df_poi, lat_col_poi, lon_col_poi)
        
        # 3. 空間結合 (sjoin_nearest)
        # 滞在点から見て、dist_threshold_m 以内で最も近いPOIを探す
        print("空間結合(Nearest Neighbor Search)を実行中...")
        
        # max_distance引数は geopandas >= 0.10.0 で利用可能
        joined = gpd.sjoin_nearest(
            gdf_stays,
            gdf_poi,
            how="left",              # マッチしなくても滞在データは残す
            max_distance=self.dist_threshold_m,
            distance_col="dist_to_poi"
        )
        
        # 4. データ整理
        # 結合できなかったデータ（Unknown）の処理
        # POI名がNaNの場所は "Unknown" 等で埋めるか、分析対象外とする
        joined['poi_name'] = joined['name'].fillna('Unknown')
        joined['poi_category'] = joined['category'].fillna('Unknown')
        
        # 緯度経度を元のWGS84に戻して保存用に整形（geometry列は削除してlat/lon列を使う）
        # ※ 元のdf_staysのlat/lonが保持されているのでそれを使う
        
        output_cols = [
            'uuid', 'stay_start_time', 'stay_end_time', 'duration_min',
            'latitude', 'longitude',  # 滞在地点の座標
            'poi_name', 'poi_category', 'dist_to_poi' # 結合されたPOI情報
        ]
        
        # 重複排除（稀に等距離で複数のPOIがヒットした場合に行が増えるのを防ぐ）
        final_df = joined[output_cols].drop_duplicates(subset=['uuid', 'stay_start_time'])
        
        # 5. 保存
        print(f"保存中...: {self.output_file}")
        final_df.to_csv(self.output_file, index=False)
        
        # 統計表示
        match_rate = (final_df['poi_name'] != 'Unknown').mean() * 100
        print(f"処理完了. POIマッチング率: {match_rate:.1f}%")
        print(final_df[['poi_name', 'poi_category', 'dist_to_poi']].head())

# ==========================================
# 実行例
# ==========================================
if __name__ == "__main__":
    
    matcher = POIMatcher(
        stay_file='./output_stays.csv',       # 前の工程で作った滞在ファイル
        poi_file='./ina_poi_data.csv',        # POIデータファイル
        output_file='./final_trip_chain.csv', # 最終出力
        dist_threshold_m=50,                  # 50m以内のPOIを紐付け
        poi_encoding='utf-8',                 # POIファイルの文字コード
        target_crs="EPSG:6676"                # 長野県(第8系)
    )
    
    matcher.process()
