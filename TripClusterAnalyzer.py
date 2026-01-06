import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMinMax

class TripClusterAnalyzer:
    def __init__(self, input_file, n_clusters=4):
        self.input_file = input_file
        self.n_clusters = n_clusters
        self.df = None
        self.model = None
        self.formatted_dataset = None
        self.feature_names = []

    def load_and_preprocess(self):
        """
        CSVを読み込み、多次元時系列データに変換する
        """
        print("データを読み込み中...")
        df = pd.read_csv(self.input_file)
        
        # 1. 時間データの数値化 (滞在開始時刻を 0.0〜24.0 の数値に変換)
        df['stay_start_time'] = pd.to_datetime(df['stay_start_time'])
        # 時間(hour) + 分(minute)/60 で数値化
        df['start_hour_numeric'] = df['stay_start_time'].dt.hour + df['stay_start_time'].dt.minute / 60.0
        
        # 2. カテゴリのOne-Hotエンコーディング
        # 'Unknown' や欠損値も1つのカテゴリとして扱う
        df['poi_category'] = df['poi_category'].fillna('Unknown')
        
        # カテゴリ変数をダミー変数化
        cat_dummies = pd.get_dummies(df['poi_category'], prefix='cat')
        df = pd.concat([df, cat_dummies], axis=1)
        
        # クラスタリングに使用する特徴量の定義
        # [緯度, 経度, 開始時刻, 滞在時間, カテゴリフラグ1, カテゴリフラグ2...]
        feature_cols = ['latitude', 'longitude', 'start_hour_numeric', 'duration_min'] + list(cat_dummies.columns)
        self.feature_names = feature_cols
        
        print(f"使用する特徴量: {len(feature_cols)}次元")
        print(feature_cols)

        # 3. ユーザーごとのシーケンス（リストのリスト）を作成
        # UUIDごとにグループ化し、時系列順に並べる
        sequences = []
        user_ids = []
        
        # 正規化のために一旦全データを取得
        raw_data = df[feature_cols].values
        
        # MinMaxスケーリング (全体で0-1に収める)
        # ※緯度経度などの比重を保つため、全体に対してscalerをfitさせる
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(raw_data)
        
        # スケーリングしたデータをDataFrameに戻してグルーピングしやすくする
        df_scaled = pd.DataFrame(scaled_data, columns=feature_cols)
        df_scaled['uuid'] = df['uuid']
        df_scaled['stay_start_time'] = df['stay_start_time'] # ソート用
        
        print("シーケンス変換中...")
        for uuid, group in df_scaled.groupby('uuid'):
            # 時系列順にソート
            group = group.sort_values('stay_start_time')
            
            # 特徴量部分のみを抽出 (uuidやtime列は除く)
            seq = group[feature_cols].values
            
            # 少なくとも2箇所以上回っている人のみを対象とする（分析の質向上のため）
            if len(seq) >= 2:
                sequences.append(seq)
                user_ids.append(uuid)
        
        self.user_ids = user_ids
        
        # tslearn用の形式に変換 (可変長データセット)
        # 異なる長さのシーケンスをNaNパディングして3次元配列にする
        self.formatted_dataset = to_time_series_dataset(sequences)
        print(f"データセット構築完了: {len(self.formatted_dataset)} ユーザー分のトリップチェーン")
        
        return df

    def run_clustering(self):
        """
        MD-DTWを用いたK-meansクラスタリングを実行
        """
        print(f"MD-DTWクラスタリングを実行中 (Clusters={self.n_clusters})...")
        print("※データ量によっては数分かかります")
        
        # metric="dtw" を指定することでDTW距離を使用
        self.model = TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            metric="dtw",
            max_iter=10,
            random_state=42,
            n_jobs=-1 # 並列処理
        )
        
        # 学習と予測
        labels = self.model.fit_predict(self.formatted_dataset)
        
        # 結果の集計
        result_df = pd.DataFrame({
            'uuid': self.user_ids,
            'cluster_id': labels
        })
        
        print("クラスタリング完了。分布:")
        print(result_df['cluster_id'].value_counts())
        
        return result_df

    def visualize_centroids(self):
        """
        各クラスターの「重心（代表的な行動パターン）」を可視化する
        緯度経度の移動だけでなく、時間やカテゴリ傾向も表示
        """
        if self.model is None:
            return

        centroids = self.model.cluster_centers_
        n_features = centroids.shape[2]
        
        # 描画設定 (緯度経度プロット + 特徴量ヒートマップ)
        fig, axes = pd.subplots(self.n_clusters, 2, figsize=(15, 4 * self.n_clusters))
        
        # 特徴量のインデックス特定
        idx_lat = self.feature_names.index('latitude')
        idx_lon = self.feature_names.index('longitude')
        idx_time = self.feature_names.index('start_hour_numeric')
        
        for i in range(self.n_clusters):
            # 重心データ (TimeStep x Features)
            center = centroids[i]
            
            # パディング(NaN)を除去（重心は固定長で出力されるが、有効な長さを見る）
            # tslearnの重心は通常最大長になるため、値の動きがある部分を見る
            
            # 左側: 軌跡プロット (Lat/Lon)
            # ※正規化されているため、0-1空間での動きになります
            ax_map = axes[i, 0]
            ax_map.plot(center[:, idx_lon], center[:, idx_lat], marker='o', linestyle='-')
            ax_map.set_title(f"Cluster {i}: Spatial Trajectory (Normalized)")
            ax_map.set_xlabel("Longitude (Scaled)")
            ax_map.set_ylabel("Latitude (Scaled)")
            ax_map.grid(True)
            
            # 始点と終点を明示
            ax_map.text(center[0, idx_lon], center[0, idx_lat], "START", color='green', fontweight='bold')
            ax_map.text(center[-1, idx_lon], center[-1, idx_lat], "END", color='red', fontweight='bold')

            # 右側: 時間とカテゴリの推移
            ax_feat = axes[i, 1]
            # 時間の推移
            ax_feat.plot(center[:, idx_time], label="Start Time", color='orange', linewidth=2)
            
            # カテゴリの強い部分を表示（値が大きい＝そのカテゴリである確率が高い）
            # カテゴリカラムのみ抽出
            cat_cols = [c for c in self.feature_names if c.startswith('cat_')]
            cat_indices = [self.feature_names.index(c) for c in cat_cols]
            
            # 各ステップで最大のカテゴリを取得して表示するのは複雑なので、
            # ここでは「カテゴリごとの平均値」を棒グラフで出す等で簡易化
            avg_cat_values = np.mean(center[:, cat_indices], axis=0)
            top_cat_idx = np.argmax(avg_cat_values)
            top_cat_name = self.feature_names[cat_indices[top_cat_idx]]
            
            ax_feat.set_title(f"Cluster {i}: Time & Main Category ({top_cat_name})")
            ax_feat.set_ylim(0, 1)
            ax_feat.legend()
            
        plt.tight_layout()
        plt.show()

# ==========================================
# 実行部
# ==========================================
if __name__ == "__main__":
    analyzer = TripClusterAnalyzer(
        input_file='final_trip_chain.csv',
        n_clusters=4 # 分類したいパターン数
    )
    
    # 1. データ読み込み・前処理
    analyzer.load_and_preprocess()
    
    # 2. クラスタリング実行
    result_df = analyzer.run_clustering()
    
    # 3. 結果の保存
    result_df.to_csv('clustering_result.csv', index=False)
    print("結果を clustering_result.csv に保存しました。")
    
    # 4. 可視化（各クラスターの特徴を確認）
    analyzer.visualize_centroids()
