import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage

class ClusterVerbalizer:
    """
    クラスタリングの数値結果（重心）を、自然言語のペルソナ記述に変換するクラス
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def verbalize(self, cluster_id, centroid_matrix):
        """
        重心行列(TimeStep x Features)を解析してテキスト化する
        ※簡易的なルールベース翻訳の実装例です
        """
        # 1. 時間帯の特徴抽出 (開始時間の平均)
        idx_time = [i for i, f in enumerate(self.feature_names) if 'start_hour' in f][0]
        avg_start_time = np.mean(centroid_matrix[:, idx_time]) * 24.0 # 0-1正規化を戻す
        
        time_desc = ""
        if avg_start_time < 11:
            time_desc = "午前中から活動を開始し、一日を有効に使います。"
        elif avg_start_time < 14:
            time_desc = "ゆっくりとしたスタートで、昼食前後から活動を始めます。"
        else:
            time_desc = "夕方や夜間の活動が中心です。"

        # 2. カテゴリの選好抽出
        # カテゴリカラムのインデックスを特定
        cat_indices = [i for i, f in enumerate(self.feature_names) if 'cat_' in f]
        
        # 重心行列全体で平均をとり、最も値が大きい（確率が高い）カテゴリを探す
        mean_cats = np.mean(centroid_matrix[:, cat_indices], axis=0)
        top_cat_idx = np.argmax(mean_cats)
        top_cat_name = self.feature_names[cat_indices[top_cat_idx]].replace('cat_', '')
        
        cat_desc = f"特に「{top_cat_name}」カテゴリの施設に強い関心があります。"
        
        # 3. 移動範囲の特徴 (緯度経度の分散などから判定可能だが今回は省略)
        move_desc = "効率的な移動を心がけます。"

        # 統合
        persona_text = (
            f"あなたは伊那市を訪れる観光客です。ID: {cluster_id} の行動パターンを持っています。\n"
            f"性格特徴: {time_desc} {cat_desc}\n"
            f"行動指針: {move_desc} 過去のデータに基づき、自分らしい選択をしてください。"
        )
        return persona_text

class TouristAgent:
    """
    LLMを搭載した観光エージェント
    """
    def __init__(self, agent_id, persona_text, api_key):
        self.agent_id = agent_id
        self.persona = persona_text
        self.current_location = "伊那市駅" # 初期位置
        self.current_time = datetime(2023, 11, 1, 10, 0) # 初期時刻
        self.history = []
        
        # LLMの初期化
        self.llm = ChatOpenAI(
            model_name="gpt-4", 
            temperature=0.7,
            openai_api_key=api_key
        )
        
        # プロンプト設計
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{persona}"),
            HumanMessagePromptTemplate.from_template(
                "現在の状況:\n"
                "- 現在地: {current_location}\n"
                "- 現在時刻: {current_time_str}\n"
                "- これまでの訪問地: {history_str}\n\n"
                "次の候補地リスト:\n{candidates_str}\n\n"
                "タスク: 上記のリストから、あなたのペルソナと現在の状況（時間・場所）に最も適した次の目的地を1つ選んでください。\n"
                "出力形式:\n"
                "決定: [施設名]\n"
                "理由: [なぜそこを選んだか、あなたの性格に基づいた理由を1文で]"
            )
        ])

    def decide_next_spot(self, candidates):
        """
        次の行き先をLLMに決定させる
        """
        # プロンプトへの入力作成
        history_str = " -> ".join(self.history) if self.history else "なし"
        candidates_str = "\n".join([f"- {c['name']} (カテゴリ: {c['category']}, 距離: {c['dist']}m)" for c in candidates])
        
        messages = self.prompt_template.format_messages(
            persona=self.persona,
            current_location=self.current_location,
            current_time_str=self.current_time.strftime("%H:%M"),
            history_str=history_str,
            candidates_str=candidates_str
        )
        
        # LLM実行
        response = self.llm.invoke(messages)
        content = response.content
        
        # 結果の解析 (簡易パーサー)
        try:
            # "決定:" の行を探す
            lines = content.split('\n')
            decision_line = [l for l in lines if l.startswith("決定:")][0]
            next_spot_name = decision_line.replace("決定:", "").strip()
            
            # 履歴更新
            self.history.append(next_spot_name)
            self.current_location = next_spot_name
            # 時間を適当に進める（実際は移動時間を計算して足す）
            self.current_time += timedelta(hours=1.5)
            
            return content
            
        except IndexError:
            return f"エラー: LLMの出力形式が不正です。\n{content}"

# ==========================================
# シミュレーション実行用モジュール
# ==========================================

def run_simulation():
    # 1. データの準備 (前の工程の結果があると仮定)
    # feature_names: 前のAnalyzerクラスの self.feature_names
    # cluster_centroids: 前のAnalyzerクラスの model.cluster_centers_
    
    # --- ダミーデータ（テスト用） ---
    print("データ準備中...")
    feature_names = ['latitude', 'longitude', 'start_hour_numeric', 'duration_min', 'cat_飲食店', 'cat_観光施設', 'cat_自然']
    
    # クラスター0の重心 (飲食店寄り)
    centroid_0 = np.array([
        [0.5, 0.5, 0.5, 0.2, 0.8, 0.1, 0.1], # Step 1
        [0.5, 0.5, 0.6, 0.2, 0.1, 0.8, 0.1]  # Step 2
    ])
    
    # 2. Verbalizerでペルソナ生成
    verbalizer = ClusterVerbalizer(feature_names)
    persona_text = verbalizer.verbalize(cluster_id=0, centroid_matrix=centroid_0)
    
    print("-" * 30)
    print("【生成されたペルソナ】")
    print(persona_text)
    print("-" * 30)
    
    # 3. エージェント生成
    # ※ここにあなたのOpenAI API Keyを入れてください
    API_KEY = "sk-..." 
    
    if API_KEY == "sk-...":
        print("API Keyが設定されていません。コード内のAPI_KEYを書き換えてください。")
        return

    agent = TouristAgent(agent_id="SimUser_01", persona_text=persona_text, api_key=API_KEY)
    
    # 4. シミュレーションステップ
    # 現在地周辺のPOI候補（本来はGeoPandasで検索して渡す）
    candidates = [
        {"name": "高遠そば ますや", "category": "飲食店", "dist": 300},
        {"name": "高遠城址公園", "category": "観光施設", "dist": 500},
        {"name": "コンビニ", "category": "小売店", "dist": 100},
        {"name": "山頂展望台", "category": "自然", "dist": 5000}
    ]
    
    print("\n【意思決定プロセス】")
    result = agent.decide_next_spot(candidates)
    print(result)

if __name__ == "__main__":
    run_simulation()
