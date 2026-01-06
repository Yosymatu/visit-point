import pandas as pd
import numpy as np
import glob
import os
import math

# ==========================================
# 設定
# ==========================================
INPUT_FOLDER = './gps_data/'        # 入力フォルダ
OUTPUT_FILE = 'stay_points_ina_city_v2.csv'

# カラム名定義 (データに合わせて変更してください)
COL_UUID = 'uuid'
COL_TIME = 'timestamp'
COL_LAT = 'latitude'
COL_LON = 'longitude'

# パラメータ設定
STAY_TIME_MIN = 15          # 15分以上で滞在とみなす
STAY_DIST_RADIUS_M = 20     # 20m以内の範囲
DRIFT_TOLERANCE_COUNT = 1   # ドリフト許容回数 (1点=2分 だけの飛び出しは無視する)

# ==========================================
# 高速な距離計算関数 (Haversine Formula)
# ==========================================
def get_dist_m(lat1, lon1, lat2, lon2):
    """
    2点間の距離(m)を計算。geopyを使わずmathで計算して高速化。
    """
    R = 6371000  # 地球半径(m)
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

# ==========================================
# 滞在抽出ロジック (ドリフト対応版)
# ==========================================
def extract_stays_robust(user_df):
    """
    ドリフト（一時的な座標飛び）を許容して滞在を判定する
    """
    # 時系列ソート
    user_df = user_df.sort_values(COL_TIME).reset_index(drop=True)
    
    # データ格納用配列（Pandasのilocは遅いのでNumpy配列で処理）
    lats = user_df[COL_LAT].values
    lons = user_df[COL_LON].values
    times = user_df[COL_TIME].values
    
    stays = []
    N = len(user_df)
    i = 0
    
    while i < N - 1:
        # 滞在候補の開始点（アンカー）
        anchor_lat = lats[i]
        anchor_lon = lons[i]
        start_time = times[i]
        
        j = i + 1
        outlier_count = 0 # 20m圏外に出た連続回数
        
        # --- 探索ループ ---
        while j < N:
            dist = get_dist_m(anchor_lat, anchor_lon, lats[j], lons[j])
            
            if dist <= STAY_DIST_RADIUS_M:
                # 20m以内: 滞在継続
                outlier_count = 0 # カウンタリセット（戻ってきたとみなす）
                
                # 【オプション】アンカーを中心位置へ徐々に更新する場合はここでanchorを平均化する
                # 今回は20mと範囲が狭いため、最初の点を固定アンカーとします
                j += 1
                
            else:
                # 20m圏外: ドリフトか移動か判定
                if outlier_count < DRIFT_TOLERANCE_COUNT:
                    # まだ許容範囲内（ドリフトとみなして無視して次を見る）
                    outlier_count += 1
                    j += 1
                else:
                    # 許容回数を超えて圏外に出た -> 本当に移動した
                    # ループを抜ける（このjは滞在に含まない）
                    # ただし、直前の数点(outlier分)も滞在から除外する必要がある
                    j -= outlier_count 
                    break
        
        # --- 滞在判定 ---
        # jは「滞在範囲を出た最初の点」のインデックス
        end_idx = j - 1
        
        if end_idx > i:
            end_time = times[end_idx]
            
            # 経過時間（分）計算
            duration_min = (end_time - start_time).astype('timedelta64[m]').astype(int)
            
            if duration_min >= STAY_TIME_MIN:
                # 滞在確定
                # 座標は期間中の平均値（ドリフト点も含めて平均するか、除外するか選べますが、ここでは含めて平均します）
                center_lat = np.mean(lats[i:j])
                center_lon = np.mean(lons[i:j])
                
                stays.append({
                    COL_UUID: user_df[COL_UUID].iloc[0],
                    'stay_start_time': start_time,
                    'stay_end_time': end_time,
                    'duration_min': duration_min,
                    'latitude': center_lat,
                    'longitude': center_lon
                })
                
                # 次の探索は、滞在終了点の次から
                i = j
            else:
                # 時間が短い -> 滞在ではない移動
                # 次の点へ（少しずつずらして探索）
                i += 1
        else:
            i += 1
            
    return stays

# ==========================================
# メイン処理
# ==========================================
def main():
    # 入力ファイル取得
    all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    print(f"Target Files: {len(all_files)}")
    
    # 出力ファイルの初期化（ヘッダーのみ書き込み）
    dummy_df = pd.DataFrame(columns=[COL_UUID, 'stay_start_time', 'stay_end_time', 'duration_min', 'latitude', 'longitude'])
    dummy_df.to_csv(OUTPUT_FILE, index=False)
    
    total_stays = 0
    
    # ファイルごとにループ処理（月またぎ考慮不要のため）
    for filepath in all_files:
        print(f"Processing: {os.path.basename(filepath)} ...")
        
        try:
            # 必要な列のみ読み込み
            df = pd.read_csv(filepath, usecols=[COL_UUID, COL_TIME, COL_LAT, COL_LON])
            df[COL_TIME] = pd.to_datetime(df[COL_TIME])
            
            # その月の滞在リスト
            monthly_stays = []
            
            # UUIDごとに処理
            # tqdmを入れると進捗が見えます: from tqdm import tqdm; for uuid, data in tqdm(df.groupby(COL_UUID)):
            for uuid, user_data in df.groupby(COL_UUID):
                if len(user_data) < 2:
                    continue
                
                stays = extract_stays_robust(user_data)
                if stays:
                    monthly_stays.extend(stays)
            
            # 結果があれば追記モードで保存
            if monthly_stays:
                result_df = pd.DataFrame(monthly_stays)
                result_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                total_stays += len(result_df)
                
            # メモリ解放
            del df
            del monthly_stays
            
        except Exception as e:
            print(f"Error in {filepath}: {e}")

    print(f"All done! Total stays extracted: {total_stays}")

if __name__ == "__main__":
    main()
