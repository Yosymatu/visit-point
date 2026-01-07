import pandas as pd
import numpy as np
import glob
import os
import math
from datetime import timedelta

class StayPointExtractor:
    def __init__(self, input_dir, output_file, 
                 dist_radius_m=20, time_min=15, drift_tolerance=1,
                 cols=None):
        """
        Args:
            cols (dict): カラム名マッピング。
                         日時が分かれている場合は 'year', 'month', 'day', 'hour', 'minute' を指定する。
        """
        self.input_dir = input_dir
        self.output_file = output_file
        self.dist_radius_m = dist_radius_m
        self.time_min = time_min
        self.drift_tolerance = drift_tolerance
        
        # デフォルトのカラム設定（ユーザー指定がない場合）
        # ※使用するCSVに合わせてここを上書きします
        default_cols = {
            'uuid': 'uuid',
            'latitude': 'latitude',
            'longitude': 'longitude',
            # 日時が分かれている場合のデフォルトキー
            'year': 'year',
            'month': 'month',
            'day': 'day',
            'hour': 'hour',
            'minute': 'minute'
        }
        self.cols = {**default_cols, **(cols or {})}

    @staticmethod
    def _calculate_distance_fast(lat1, lon1, lat2, lon2):
        """Haversine formulaによる高速距離計算(m)"""
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def _extract_from_trajectory(self, user_df):
        """
        1ユーザー分の軌跡データから滞在リストを抽出する
        """
        lats = user_df[self.cols['latitude']].values
        lons = user_df[self.cols['longitude']].values
        # ここでは変換済みの 'timestamp' 列（datetime型）を使用
        times = user_df['timestamp'].values 
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
            
            while j < N:
                dist = self._calculate_distance_fast(anchor_lat, anchor_lon, lats[j], lons[j])
                
                if dist <= self.dist_radius_m:
                    outlier_count = 0 
                    j += 1
                else:
                    if outlier_count < self.drift_tolerance:
                        outlier_count += 1
                        j += 1
                    else:
                        j -= outlier_count
                        break
            
            end_idx = j - 1
            if end_idx > i:
                end_time = times[end_idx]
                duration_min = (end_time - start_time).astype('timedelta64[m]').astype(int)
                
                if duration_min >= self.time_min:
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
        """メイン処理"""
        self._initialize_output_file()
        files = sorted(glob.glob(os.path.join(self.input_dir, "*.csv")))
        print(f"Target Files: {len(files)}")
        
        for filepath in files:
            print(f"Processing: {os.path.basename(filepath)} ...")
            try:
                self._process_single_file(filepath)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                import traceback
                traceback.print_exc()

    def _initialize_output_file(self):
        header_df = pd.DataFrame(columns=[
            'uuid', 'stay_start_time', 'stay_end_time', 
            'duration_min', 'latitude', 'longitude'
        ])
        header_df.to_csv(self.output_file, index=False)

    def _process_single_file(self, filepath):
        """
        単一ファイルの読み込み・日時結合・処理
        """
        # 1. 読み込むべきカラムのリストを作成
        # UUID, Lat, Lon に加えて、日時構成カラムすべてを指定
        date_components = ['year', 'month', 'day', 'hour', 'minute']
        target_cols = [self.cols[k] for k in date_components] + \
                      [self.cols['uuid'], self.cols['latitude'], self.cols['longitude']]
        
        # CSV読み込み
        df = pd.read_csv(filepath, usecols=target_cols)
        
        # 2. pd.to_datetimeを使って日時列を一本化
        # to_datetimeは、'year', 'month', ... という正確な列名を持つDataFrameを渡すと変換してくれる
        
        # 日時カラムだけを抜き出し、Pandasが要求する英語名にリネームする辞書を作成
        rename_map = {self.cols[k]: k for k in date_components}
        
        # 一時的なDataFrameを作って変換（元のdfを壊さないため）
        temp_date_df = df[list(rename_map.keys())].rename(columns=rename_map)
        
        # 結合して新しい 'timestamp' 列を作成
        # errors='coerce' で無効な日付はNaTにする
        df['timestamp'] = pd.to_datetime(temp_date_df, errors='coerce')
        
        # 変換に失敗した行（NaT）があれば削除
        if df['timestamp'].isna().any():
            print(f"  Warning: Invalid dates found and dropped.")
            df = df.dropna(subset=['timestamp'])

        # 3. ユーザーごとに処理
        for _, user_data in df.groupby(self.cols['uuid']):
            user_data = user_data.sort_values('timestamp')
            if len(user_data) < 2:
                continue
                
            stays = self._extract_from_trajectory(user_data)
            if stays:
                # 抽出された結果をCSVへ追記
                result_df = pd.DataFrame(stays)
                result_df.to_csv(self.output_file, mode='a', header=False, index=False)

        # メモリ解放
        del df

# ==========================================
# 実行例（カラム設定例）
# ==========================================
if __name__ == "__main__":
    
    extractor = StayPointExtractor(
        input_dir='./gps_data_monthly/',
        output_file='./output_stays.csv',
        dist_radius_m=20,
        time_min=15,
        drift_tolerance=1,
        
        # 【重要】実際のCSVのヘッダー名に合わせてここを設定してください
        cols={
            'uuid': 'uuid',          # CSV内のID列名
            'latitude': 'latitude',  # CSV内の緯度列名
            'longitude': 'longitude',# CSV内の経度列名
            
            # 日時が分かれている列名
            'year': 'year',     # 例: 'YYYY' なら 'year': 'YYYY' と書く
            'month': 'month',   # 例: 'MM'
            'day': 'day',       # 例: 'DD'
            'hour': 'hour',     # 例: 'hh'
            'minute': 'minute'  # 例: 'mm'
        }
    )
    
    extractor.process_files()
