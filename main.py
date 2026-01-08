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
    
    # 1. Stay Extraction
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

    # 2. POI Matching
    if not os.path.exists(FINAL_CHAIN):
        print("Starting POI Matching...")
        matcher = POIMatcher(INTERMEDIATE, POI_FILE, FINAL_CHAIN)
        matcher.process()

    # 3. Clustering
    analyzer = TripClusterAnalyzer(FINAL_CHAIN, n_clusters=3)
    centroids, features = analyzer.analyze()
    
    if centroids is None:
        print("Clustering failed due to insufficient data.")
        return

    # ==========================================================
    # ここが変更点: クラスター特徴の説明を表示
    # ==========================================================
    print("\n[Phase 4] Analyzing Cluster Characteristics...")
    
    verbalizer = ClusterVerbalizerEn(features)
    # 全クラスターの特徴を表示
    summaries = verbalizer.explain_all_clusters(centroids)
    
    # シミュレーションしたいクラスターIDを指定（ここでは例として0番）
    # ※ 実際にはユーザー入力(input())で選ばせても良いです
    target_cluster_id = 0
    print(f"\nRunning Simulation for -> Cluster {target_cluster_id}")
    
    # ペルソナ生成
    persona = verbalizer.verbalize(target_cluster_id, centroids[target_cluster_id])
    print(f"--- Generated Persona ---\n{persona}\n-------------------------")
    
    # ==========================================================
    
    # 4. Simulation
    searcher = POISearcher(POI_FILE)
    agent = TouristAgentEn(persona, model_name="llama3")
    
    start_spot = "伊那市駅"
    current_lat, current_lon = searcher.get_coords_by_name(start_spot) or (35.8398, 137.9622)
    agent.current_location_name = start_spot
    print(f"Start: {start_spot} ({current_lat:.4f}, {current_lon:.4f})")

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
