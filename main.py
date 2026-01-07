import os
from stay_extractor import StayPointExtractor
from poi_matcher import POIMatcher
from analyzer import TripClusterAnalyzer
from poi_searcher import POISearcher
from agent import ClusterVerbalizerEn, TouristAgentEn

def main():
    # パス設定
    GPS_DIR = './gps_data_monthly/'
    INTERMEDIATE = './intermediate_stays.csv'
    POI_FILE = './ina_buildings.geojson'
    FINAL_CHAIN = './ina_trip_chains.csv'
    
    # 1. 滞在抽出 (Phase 1)
    if not os.path.exists(INTERMEDIATE):
        print("Starting Stay Extraction...")
        extractor = StayPointExtractor(
            GPS_DIR, INTERMEDIATE, 
            cols={
                'uuid':'uuid', 'latitude':'latitude', 'longitude':'longitude',
                'year':'year', 'month':'month', 'day':'day', 'hour':'hour', 'minute':'minute'
            }
        )
        extractor.process_files()

    # 2. POI結合 (Phase 2)
    if not os.path.exists(FINAL_CHAIN):
        print("Starting POI Matching...")
        matcher = POIMatcher(INTERMEDIATE, POI_FILE, FINAL_CHAIN)
        matcher.process()

    # 3. クラスタリング分析 (Phase 3)
    analyzer = TripClusterAnalyzer(FINAL_CHAIN, n_clusters=3)
    centroids, features = analyzer.analyze()
    
    if centroids is None:
        print("Clustering failed due to insufficient data.")
        return

    # 4. エージェントシミュレーション (Phase 4)
    print("\n[Phase 4] Running Simulation...")
    
    # 検索クラスの準備
    searcher = POISearcher(POI_FILE)
    
    # クラスタ0のペルソナ生成
    verbalizer = ClusterVerbalizerEn(features)
    persona = verbalizer.verbalize(0, centroids[0])
    print(f"--- Persona ---\n{persona}\n---------------")
    
    # エージェント初期化
    agent = TouristAgentEn(persona, model_name="llama3")
    
    # 開始地点の設定（例：伊那市駅）
    start_spot = "伊那市駅"
    current_lat, current_lon = searcher.get_coords_by_name(start_spot) or (35.8398, 137.9622)
    agent.current_location_name = start_spot
    print(f"Start: {start_spot} ({current_lat:.4f}, {current_lon:.4f})")

    # 意思決定ループ（1回分）
    candidates = searcher.search_nearby(current_lat, current_lon, radius_m=2000, limit=5)
    
    if candidates:
        full_response, decision_name = agent.decide(candidates)
        print(f"\n{full_response}\n")
        
        next_coords = searcher.get_coords_by_name(decision_name)
        if next_coords:
            print(f"Agent moved to: {decision_name}")
    else:
        print("No candidates found.")

if __name__ == "__main__":
    main()