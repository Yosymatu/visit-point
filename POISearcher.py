import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

class POISearcher:
    """
    GeoPandasを用いて現在地周辺のPOIを検索するクラス
    """
    def __init__(self, poi_csv_file, target_crs="EPSG:6676"):
        """
        Args:
            poi_csv_file (str): POIデータのCSVパス
            target_crs (str): 距離計算用の投影座標系 (長野はEPSG:6676)
        """
        # 1. データ読み込み
        df = pd.read_csv(poi_csv_file)
        
        # 緯度経度列の名前解決（適宜調整してください）
        lat_col = 'latitude' if 'latitude' in df.columns else 'lat'
        lon_col = 'longitude' if 'longitude' in df.columns else 'lon'
        
        # 2. GeoDataFrame化 (WGS84)
        self.gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326"
        )
        
        # 3. 距離計算用に投影変換しておく (高速化のためinitでやっておく)
        self.gdf_projected = self.gdf.to_crs(target_crs)
        self.target_crs = target_crs

    def search_nearby(self, current_lat, current_lon, radius_m=2000, limit=10):
        """
        指定座標から半径m以内のPOIを検索してリストで返す
        """
        # 現在地をPointオブジェクト化
        current_point = gpd.GeoSeries(
            [Point(current_lon, current_lat)], 
            crs="EPSG:4326"
        )
        
        # 距離計算用座標系へ変換
        current_point_proj = current_point.to_crs(self.target_crs).iloc[0]
        
        # --- 距離計算 & フィルタリング ---
        # 全POIとの距離を計算
        distances = self.gdf_projected.distance(current_point_proj)
        
        # 半径以内のインデックスを取得
        mask = distances <= radius_m
        nearby_pois = self.gdf_projected[mask].copy()
        
        # 距離列を追加
        nearby_pois['dist'] = distances[mask]
        
        # 近い順にソートして、上位limit件に絞る
        nearby_pois = nearby_pois.sort_values('dist').head(limit)
        
        # LLMに渡すための辞書リスト形式に変換
        # 必要なカラム: name, category, dist
        candidates = []
        for _, row in nearby_pois.iterrows():
            # 自分自身（距離0m）は除外する場合
            if row['dist'] < 1.0: 
                continue
                
            candidates.append({
                "name": row['name'],          # CSVの施設名カラム
                "category": row['category'],  # CSVのカテゴリカラム
                "dist": int(row['dist'])      # 整数メートル
            })
            
        return candidates

    def get_coords_by_name(self, spot_name):
        """
        施設名から座標(lat, lon)を取得するヘルパー関数
        （エージェントが次の場所に移動した際、その座標を知るために使用）
        """
        # 完全一致検索 (必要に応じて部分一致などに変更)
        target = self.gdf[self.gdf['name'] == spot_name]
        if len(target) > 0:
            point = target.iloc[0].geometry
            return point.y, point.x  # lat, lon
        return None

# ==========================================
# 更新されたシミュレーション実行部
# ==========================================
def run_simulation_with_geopandas():
    # 1. 検索クラスの初期化 (事前にロードしておく)
    # ※ POIファイルには name, category, latitude, longitude 列が必要です
    poi_searcher = POISearcher(poi_csv_file='ina_poi_data.csv')
    
    # ... (前略: エージェント初期化などは同じ) ...
    
    # 2. エージェントの現在地の座標を設定
    # 初期位置（例: 伊那市駅）の座標を取得
    current_spot_name = "伊那市駅"
    current_coords = poi_searcher.get_coords_by_name(current_spot_name)
    
    if current_coords is None:
        # 見つからない場合はデフォルト値やエラー処理
        current_lat, current_lon = 35.8398, 137.9622 
    else:
        current_lat, current_lon = current_coords

    print(f"現在地: {current_spot_name} ({current_lat:.4f}, {current_lon:.4f})")
    
    # ----------------------------------------------------
    # 3. ここが変更点: GeoPandasで動的に検索して candidates を作成
    # ----------------------------------------------------
    candidates = poi_searcher.search_nearby(
        current_lat=current_lat, 
        current_lon=current_lon, 
        radius_m=3000, # 半径3km以内
        limit=10       # 上位10件
    )
    
    # 検索結果が空の場合の対策（範囲を広げるか、タクシー等を出すか）
    if not candidates:
        print("近くに施設が見つかりませんでした。検索範囲を広げます。")
        candidates = poi_searcher.search_nearby(current_lat, current_lon, radius_m=10000, limit=5)

    print("\n【検索された周辺候補】")
    for c in candidates:
        print(f"- {c['name']} ({c['category']}): {c['dist']}m")

    # 4. エージェントによる意思決定 (ここは前回と同じ)
    # result = agent.decide_next_spot(candidates)
    # print(result)
    
    # --- 移動後の処理イメージ ---
    # エージェントが「高遠城址公園」を選んだとしたら...
    # next_spot_name = "高遠城址公園" (LLMの出力から抽出)
    # next_coords = poi_searcher.get_coords_by_name(next_spot_name)
    # if next_coords:
    #     current_lat, current_lon = next_coords
    #     # ループの次へ...

if __name__ == "__main__":
    # 事前にダミーCSVがないと動かないため、try-exceptで囲むか、ファイルを準備してください
    try:
        run_simulation_with_geopandas()
    except FileNotFoundError:
        print("エラー: 'ina_poi_data.csv' が見つかりません。POIデータを用意してください。")
