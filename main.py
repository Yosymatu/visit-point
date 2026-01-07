import os

# ========================================================
# ここにこれまでに作成した2つのクラス定義を貼り付けるか、
# 別ファイル(ex: modules.py)からimportしてください。
# from modules import StayPointExtractor, POIMatcher
# ========================================================

def generate_trip_chains_pipeline():
    print("=== Phase 1: GPSログからの滞在点抽出 (Stay Point Extraction) ===")
    
    # 一時ファイルのパス
    intermediate_stay_file = './intermediate_stays.csv'
    
    # 1. 滞在抽出器の初期化
    extractor = StayPointExtractor(
        input_dir='./gps_data_monthly/',       # 月次GPSデータのフォルダ
        output_file=intermediate_stay_file,    # 中間出力ファイル
        dist_radius_m=20,                      # 20m以内の範囲
        time_min=15,                           # 15分以上の滞在
        drift_tolerance=1,                     # ドリフト許容(1回)
        
        # 【重要】お手元のCSVカラム名に合わせてマッピング
        cols={
            'uuid': 'uuid',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'year': 'year',      # 日時が分かれている場合
            'month': 'month',
            'day': 'day',
            'hour': 'hour',
            'minute': 'minute'
        }
    )
    
    # 実行
    extractor.process_files()
    
    print("\n=== Phase 2: 建物ポリゴンとの空間結合 (POI Matching) ===")
    
    # 最終出力ファイルのパス
    final_output_file = './ina_trip_chains.csv'
    
    # 2. POIマッチャーの初期化
    matcher = POIMatcher(
        stay_file=intermediate_stay_file,         # Phase 1の出力
        poi_geojson_file='./ina_buildings.geojson', # 建物ポリゴンデータ
        output_file=final_output_file,            # 最終的なトリップチェーン
        dist_threshold_m=30,                      # 建物から30m以内なら滞在とみなす
        target_crs="EPSG:6676"                    # 長野県の座標系
    )
    
    # 実行
    matcher.process()

    print(f"\n=== 全工程完了 ===")
    print(f"生成されたトリップチェーン: {final_output_file}")

if __name__ == "__main__":
    # 事前チェック: 入力データがあるか確認
    if not os.path.exists('./gps_data_monthly/'):
        print("エラー: './gps_data_monthly/' フォルダが見つかりません。")
    elif not os.path.exists('./ina_buildings.geojson'):
        print("エラー: './ina_buildings.geojson' が見つかりません。")
    else:
        generate_trip_chains_pipeline()