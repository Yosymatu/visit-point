import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

class TripClusterAnalyzer:
    def __init__(self, input_file, n_clusters=3):
        self.input_file = input_file
        self.n_clusters = n_clusters

    def analyze(self):
        print(f"[Phase 3] Clustering Trip Chains...")
        df = pd.read_csv(self.input_file)
        
        # Unknown除去
        df = df[df['poi_category'] != 'Unknown'].copy()
        if len(df) == 0: return None, None
        
        # 特徴量作成
        df['stay_start_time'] = pd.to_datetime(df['stay_start_time'])
        df['start_hour_numeric'] = df['stay_start_time'].dt.hour + df['stay_start_time'].dt.minute / 60.0
        cat_dummies = pd.get_dummies(df['poi_category'], prefix='cat')
        
        feature_cols = ['latitude', 'longitude', 'start_hour_numeric', 'duration_min'] + list(cat_dummies.columns)
        
        # 正規化
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols].values)
        df_scaled = pd.DataFrame(scaled_data, columns=feature_cols)
        df_scaled['uuid'] = df['uuid'].values
        df_scaled['time'] = df['stay_start_time'].values

        # シーケンス作成
        sequences = []
        for _, group in df_scaled.groupby('uuid'):
            seq = group.sort_values('time')[feature_cols].values
            if len(seq) >= 2:
                sequences.append(seq)
        
        if not sequences:
            print("No valid sequences found.")
            return None, None
            
        model = TimeSeriesKMeans(n_clusters=self.n_clusters, metric="dtw", max_iter=5, random_state=42)
        model.fit(to_time_series_dataset(sequences))
        
        return model.cluster_centers_, feature_cols