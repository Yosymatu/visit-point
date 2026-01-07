import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

class POISearcher:
    """
    GeoJSON(Polygon)を用いて現在地周辺のPOIを検索するクラス
    """
    def __init__(self, poi_geojson_file, target_crs="EPSG:6676"):
        # 1. GeoJSON読み込み
        self.gdf = gpd.read_file(poi_geojson_file)
        
        # 2. 距離計算用に投影変換
        self.gdf_projected = self.gdf.to_crs(target_crs)
        self.target_crs = target_crs
        
        # 3. エージェントの移動先座標として使うため、あらかじめ「重心」を計算しておく
        # (Polygonのままだと lat, lon が一意に決まらないため)
        self.gdf_projected['centroid_geom'] = self.gdf_projected.geometry.centroid
        
        # 重心をWGS84に戻して lat/lon 列を作っておく（LLMへの提示やログ用）
        centroids_wgs84 = self.gdf_projected['centroid_geom'].to_crs("EPSG:4326")
        self.gdf_projected['center_lat'] = centroids_wgs84.y
        self.gdf_projected['center_lon'] = centroids_wgs84.x

    def search_nearby(self, current_lat, current_lon, radius_m=2000, limit=10):
        """
        指定座標から半径m以内の建物POIを検索
        """
        current_point = gpd.GeoSeries(
            [Point(current_lon, current_lat)], 
            crs="EPSG:4326"
        ).to_crs(self.target_crs).iloc[0]
        
        # Polygonとの距離計算 (Polygon内なら0になる)
        distances = self.gdf_projected.geometry.distance(current_point)
        
        mask = distances <= radius_m
        nearby_pois = self.gdf_projected[mask].copy()
        nearby_pois['dist'] = distances[mask]
        
        nearby_pois = nearby_pois.sort_values('dist').head(limit)
        
        candidates = []
        for _, row in nearby_pois.iterrows():
            # GeoJSONのプロパティ名に合わせてください
            name = row.get('name', 'Unknown Building') 
            category = row.get('category', 'Unknown')
            
            candidates.append({
                "name": name,
                "category": category,
                "dist": int(row['dist'])
            })
        return candidates

    def get_coords_by_name(self, spot_name):
        """
        施設名から「重心座標」を取得する
        """
        target = self.gdf_projected[self.gdf_projected['name'] == spot_name]
        if len(target) > 0:
            # 事前に計算しておいた重心を使用
            lat = target.iloc[0]['center_lat']
            lon = target.iloc[0]['center_lon']
            return lat, lon
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
