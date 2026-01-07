import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

class TripClusterAnalyzer:
    def __init__(self, input_file, n_clusters=4):
        self.input_file = input_file
        self.n_clusters = n_clusters
        self.df = None
        self.model = None
        self.formatted_dataset = None
        self.feature_names = []
        self.user_ids = []

    def load_and_preprocess(self):
        """
        CSVを読み込み、Unknownカテゴリを除外した上で多次元時系列データに変換する
        """
        print("データを読み込み中...")
        df = pd.read_csv(self.input_file)
        
        # ===========================================================
        # 修正点: 'Unknown' カテゴリの除外処理
        # ===========================================================
        # まず欠損値をUnknownで埋める（念のため）
        df['poi_category'] = df['poi_category'].fillna('Unknown')
        
        # Unknownを除外
        initial_len = len(df)
        df = df[df['poi_category'] != 'Unknown'].copy()
        dropped_len = initial_len - len(df)
        print(f"前処理: 'Unknown' カテゴリの滞在 {dropped_len} 件を除外しました。")
        
        if len(df) == 0:
            raise ValueError("有効なカテゴリを持つデータがありません。POI結合の設定を確認してください。")

        # 1. 時間データの数値化
        df['stay_start_time'] = pd.to_datetime(df['stay_start_time'])
        df['start_hour_numeric'] = df['stay_start_time'].dt.hour + df['stay_start_time'].dt.minute / 60.0
        
        # 2. カテゴリのOne-Hotエンコーディング
        # Unknownを除去したので、残っているカテゴリだけでダミー変数化
        cat_dummies = pd.get_dummies(df['poi_category'], prefix='cat')
        df = pd.concat([df, cat_dummies], axis=1)
        
        # 特徴量の定義
        feature_cols = ['latitude', 'longitude', 'start_hour_numeric', 'duration_min'] + list(cat_dummies.columns)
        self.feature_names = feature_cols
        
        print(f"使用する特徴量 ({len(feature_cols)}次元): {feature_cols}")

        # 3. ユーザーごとのシーケンス作成
        sequences = []
        user_ids = []
        
        # 正規化 (MinMax Scaling)
        raw_data = df[feature_cols].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(raw_data)
        
        # スケーリング後のデータをDataFrameに戻す
        df_scaled = pd.DataFrame(scaled_data, columns=feature_cols)
        df_scaled['uuid'] = df['uuid'].values
        df_scaled['stay_start_time'] = df['stay_start_time'].values
        
        print("トリップチェーン構築中...")
        # UUIDごとにグループ化
        for uuid, group in df_scaled.groupby('uuid'):
            # 時系列順にソート
            group = group.sort_values('stay_start_time')
            
            # 特徴量のみ抽出
            seq = group[feature_cols].values
            
            # 【重要】Unknownを除去した結果、滞在地点が2箇所以上のユーザーのみ対象にする
            # (1箇所だけだと「移動（チェーン）」の分析ができないため)
            if len(seq) >= 2:
                sequences.append(seq)
                user_ids.append(uuid)
        
        self.user_ids = user_ids
        
        if len(sequences) == 0:
            raise ValueError("条件を満たす（2箇所以上滞在した）ユーザーがいません。")

        # tslearn用に変換
        self.formatted_dataset = to_time_series_dataset(sequences)
        print(f"データセット構築完了: {len(self.formatted_dataset)} ユーザー分のトリップチェーン (Unknown除外済み)")
        
        return df

    def run_clustering(self):
        """MD-DTWを用いたK-meansクラスタリングを実行"""
        if self.formatted_dataset is None:
            print("エラー: 先に load_and_preprocess() を実行してください。")
            return None

        print(f"MD-DTWクラスタリングを実行中 (Clusters={self.n_clusters})...")
        
        self.model = TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            metric="dtw",
            max_iter=10,
            random_state=42,
            n_jobs=-1
        )
        
        labels = self.model.fit_predict(self.formatted_dataset)
        
        result_df = pd.DataFrame({
            'uuid': self.user_ids,
            'cluster_id': labels
        })
        
        print("クラスタリング完了。分布:")
        print(result_df['cluster_id'].value_counts())
        
        return result_df

    def visualize_centroids(self):
        """重心の可視化"""
        if self.model is None:
            return

        centroids = self.model.cluster_centers_
        
        # 可視化設定
        fig, axes = plt.subplots(self.n_clusters, 2, figsize=(15, 4 * self.n_clusters))
        
        # インデックス特定
        idx_lat = self.feature_names.index('latitude')
        idx_lon = self.feature_names.index('longitude')
        idx_time = self.feature_names.index('start_hour_numeric')
        cat_cols = [c for c in self.feature_names if c.startswith('cat_')]
        cat_indices = [self.feature_names.index(c) for c in cat_cols]

        for i in range(self.n_clusters):
            center = centroids[i]
            
            # 左: 軌跡
            ax_map = axes[i, 0] if self.n_clusters > 1 else axes[0]
            ax_map.plot(center[:, idx_lon], center[:, idx_lat], marker='o', linestyle='-')
            ax_map.set_title(f"Cluster {i}: Trajectory (Lat/Lon)")
            ax_map.grid(True)
            ax_map.text(center[0, idx_lon], center[0, idx_lat], "START", color='green', fontweight='bold')
            ax_map.text(center[-1, idx_lon], center[-1, idx_lat], "END", color='red', fontweight='bold')

            # 右: 時間とカテゴリ
            ax_feat = axes[i, 1] if self.n_clusters > 1 else axes[1]
            ax_feat.plot(center[:, idx_time], label="Time", color='orange', linewidth=2, linestyle='--')
            
            # カテゴリ傾向（重心の平均値）
            avg_cat_values = np.mean(center[:, cat_indices], axis=0)
            
            # 棒グラフでカテゴリを表示
            # x軸の設定が煩雑になるため、ここでは主要カテゴリ名を表示するのみとします
            top_cat_idx = np.argmax(avg_cat_values)
            top_cat_name = self.feature_names[cat_indices[top_cat_idx]].replace('cat_', '')
            
            ax_feat.bar(range(len(cat_cols)), avg_cat_values, alpha=0.6, label='Category Prob')
            ax_feat.set_xticks(range(len(cat_cols)))
            ax_feat.set_xticklabels([c.replace('cat_', '') for c in cat_cols], rotation=45, ha='right')
            
            ax_feat.set_title(f"Cluster {i}: Time & Top Category ({top_cat_name})")
            ax_feat.legend()

        plt.tight_layout()
        plt.show()

# ==========================================
# 実行部
# ==========================================
if __name__ == "__main__":
    # ポリゴン結合後のCSV（final_trip_chain_poly.csv）を入力とする想定
    analyzer = TripClusterAnalyzer(
        input_file='final_trip_chain_poly.csv',
        n_clusters=24
    )
    
    try:
        analyzer.load_and_preprocess()
        result = analyzer.run_clustering()
        if result is not None:
            result.to_csv('clustering_result.csv', index=False)
            analyzer.visualize_centroids()
    except Exception as e:
        print(f"エラーが発生しました: {e}")