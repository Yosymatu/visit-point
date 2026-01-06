import pandas as pd
import numpy as np
import glob
import os
import math
from datetime import timedelta

class StayPointExtractor:
    """
    GPS軌跡データから滞在地点(Stay Point)を抽出するクラス。
    ドリフト（一時的な座標飛び）への許容ロジックを含む。
    """

    def __init__(self, input_dir, output_file, 
                 dist_radius_m=20, time_min=15, drift_tolerance=1,
                 cols=None):
        """
        Args:
            input_dir (str): 入力CSVファイルがあるフォルダパス
            output_file (str): 出力するCSVのファイルパス
            dist_radius_m (int): 滞在とみなす判定円の半径(m)
            time_min (int): 滞在とみなす最小時間(分)
            drift_tolerance (int): 判定円を出ても許容する回数（ドリフト対策）。1=2分間の飛び出しを許容。
            cols (dict): CSVのカラム名マッピング
        """
        self.input_dir = input_dir
        self.output_file = output_file
        self.dist_radius_m = dist_radius_m
        self.time_min = time_min
        self.drift_tolerance = drift_tolerance
        
        # デフォルトのカラム設定
        default_cols = {
            'uuid': 'uuid',
            'timestamp': 'timestamp',
            'latitude': 'latitude',
            'longitude': 'longitude'
        }
        self.cols = {**default_cols, **(cols or {})}

    @staticmethod
    def _calculate_distance_fast(lat1, lon1, lat2, lon2):
        """
        2点間の距離(m)を計算する静的メソッド (Haversine formula)
        """
        R = 6371000  # 地球半径(m)
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def _extract_from_trajectory(self, user_df):
        """
        1ユーザー分のデータフレームから滞在リストを抽出するコアロジック
        """
        # Numpy配列化して高速アクセス
        lats = user_df[self.cols['latitude']].values
        lons = user_df[self.cols['longitude']].values
        times = user_df[self.cols['timestamp']].values
        uuid = user_df[self.cols['uuid']].iloc[0]
        
        stays = []
        N = len(user_df)
        i = 0
        
        while i < N - 1:
            anchor_lat = lats[i]
            anchor_lon = lons[i]
            start_time = times[i]
            
            j = i + 1
            outlier_count = 0
            
            # --- 探索ループ ---
            while j < N:
                dist = self._calculate_distance_fast(anchor_lat, anchor_lon, lats[j], lons[j])
                
                if dist <= self.dist_radius_m:
                    # 滞在範囲内
                    outlier_count = 0 
                    j += 1
                else:
                    # 範囲外（ドリフト判定）
                    if outlier_count < self.drift_tolerance:
                        outlier_count += 1
                        j += 1
                    else:
                        # 許容回数を超えたため移動とみなす
                        # outlier分巻き戻してループを抜ける
                        j -= outlier_count 
                        break
            
            # --- 滞在判定 ---
            end_idx = j - 1
            if end_idx > i:
                end_time = times[end_idx]
                # timedelta64[ns] -> 分に変換
                duration_min = (end_time - start_time).astype('timedelta64[m]').astype(int)
                
                if duration_min >= self.time_min:
                    # 滞在確定: 重心を計算
                    # (ドリフト点も含めて平均化する場合)
                    center_lat = np.mean(lats[i:j])
                    center_lon = np.mean(lons[i:j])
                    
                    stays.append({
                        'uuid': uuid,
                        'stay_start_time': start_time,
                        'stay_end_time': end_time,
                        'duration_min': duration_min,
                        'latitude': center_lat,
                        'longitude': center_lon
                    })
                    i = j
                else:
                    i += 1
            else:
                i += 1
                
        return stays

    def process_files(self):
        """
        フォルダ内の全ファイルを順次処理してCSVに出力するメインメソッド
        """
        # 出力ファイルの初期化
        self._initialize_output_file()
        
        files = glob.glob(os.path.join(self.input_dir, "*.csv"))
        print(f"Target Files: {len(files)}")
        
        total_stays_count = 0
        
        for filepath in files:
            print(f"Processing: {os.path.basename(filepath)} ...")
            try:
                self._process_single_file(filepath)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                
        print(f"Processing complete.")

    def _initialize_output_file(self):
        """出力ファイルのヘッダー作成"""
        header_df = pd.DataFrame(columns=[
            'uuid', 'stay_start_time', 'stay_end_time', 
            'duration_min', 'latitude', 'longitude'
        ])
        header_df.to_csv(self.output_file, index=False)

    def _process_single_file(self, filepath):
        """単一ファイルの読み込み・処理・追記保存"""
        # 必要な列のみ読み込み
        usecols = list(self.cols.values())
        df = pd.read_csv(filepath, usecols=usecols)
        df[self.cols['timestamp']] = pd.to_datetime(df[self.cols['timestamp']])
        
        monthly_stays = []
        
        # ユーザーごとに処理
        for _, user_data in df.groupby(self.cols['uuid']):
            user_data = user_data.sort_values(self.cols['timestamp'])
            if len(user_data) < 2:
                continue
                
            stays = self._extract_from_trajectory(user_data)
            if stays:
                monthly_stays.extend(stays)
        
        # 結果があれば追記
        if monthly_stays:
            result_df = pd.DataFrame(monthly_stays)
            result_df.to_csv(self.output_file, mode='a', header=False, index=False)


# ==========================================
# 実行例
# ==========================================
if __name__ == "__main__":
    
    # 1. インスタンス生成
    extractor = StayPointExtractor(
        input_dir='./gps_data/',              # 入力フォルダ
        output_file='./output_stays.csv',     # 出力ファイル
        dist_radius_m=20,                     # 半径20m
        time_min=15,                          # 15分以上
        drift_tolerance=1,                    # ドリフト許容(1点)
        cols={                                # カラム名マッピング（任意）
            'uuid': 'uuid',
            'timestamp': 'timestamp',
            'latitude': 'latitude',
            'longitude': 'longitude'
        }
    )
    
    # 2. 実行
    extractor.process_files()
