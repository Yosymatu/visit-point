import numpy as np
from datetime import datetime
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

class ClusterVerbalizerEn:
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def _analyze_single_cluster(self, centroid_matrix):
        """
        単一の重心行列から特徴（時間、滞在時間、カテゴリ）を抽出するヘルパーメソッド
        """
        # 1. 時間帯の傾向 (Start Hour)
        idx_time = [i for i, f in enumerate(self.feature_names) if 'start_hour' in f][0]
        # 正規化されているため 0.0(データ内の最早)〜1.0(最遅)
        avg_time_score = np.mean(centroid_matrix[:, idx_time])
        
        if avg_time_score < 0.3:
            time_type = "Early Morning"
        elif avg_time_score < 0.6:
            time_type = "Daytime"
        else:
            time_type = "Late Afternoon / Night"

        # 2. 滞在時間の傾向 (Duration)
        idx_dur = [i for i, f in enumerate(self.feature_names) if 'duration' in f][0]
        avg_dur_score = np.mean(centroid_matrix[:, idx_dur])
        
        if avg_dur_score < 0.3:
            dur_type = "Short stays (Quick visits)"
        elif avg_dur_score < 0.6:
            dur_type = "Medium length stays"
        else:
            dur_type = "Long stays (Deep exploration)"

        # 3. メインカテゴリの特定
        cat_indices = [i for i, f in enumerate(self.feature_names) if 'cat_' in f]
        if cat_indices:
            mean_cats = np.mean(centroid_matrix[:, cat_indices], axis=0)
            top_cat_idx = np.argmax(mean_cats)
            # 'cat_' を除去してカテゴリ名を取得
            top_cat = self.feature_names[cat_indices[top_cat_idx]].replace('cat_', '')
        else:
            top_cat = "General"

        return time_type, dur_type, top_cat

    def explain_all_clusters(self, centroids):
        """
        全クラスターの特徴を要約して辞書で返す
        """
        summaries = {}
        print("\n=== Cluster Characteristics Analysis ===")
        for i, center in enumerate(centroids):
            time_type, dur_type, top_cat = self._analyze_single_cluster(center)
            
            desc = (f"[Cluster {i}] is {time_type} type. "
                    f"Prefers '{top_cat}'. "
                    f"Tendency: {dur_type}.")
            
            summaries[i] = desc
            print(desc)
        print("========================================\n")
        return summaries

    def verbalize(self, cluster_id, centroid_matrix):
        """
        特定クラスターのペルソナ（System Prompt用）を生成
        """
        time_type, dur_type, top_cat = self._analyze_single_cluster(centroid_matrix)
        
        return (f"You are a tourist visiting Ina City. Your ID is {cluster_id}.\n"
                f"Personality: You act mainly during {time_type}. "
                f"You strongly prefer '{top_cat}' and usually make {dur_type}.\n"
                f"Guideline: Make decisions that reflect this personality.")

# --- TouristAgentEn クラスは変更なし ---
import random

class TouristAgentEn:
    def __init__(self, persona_text, model_name="llama3"):
        self.persona = persona_text
        self.current_location_name = "Start Point"
        self.current_time = datetime(2023, 11, 1, 10, 0)
        self.history = []
        self.llm = ChatOllama(model=model_name, temperature=0.7) # ここは0.7でOK
        
        # プロンプトを変更: 1つ選ぶのではなく、スコア(0-10)を付けさせる
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{persona}"),
            HumanMessagePromptTemplate.from_template(
                "Situation:\n- Location: {loc}\n- Time: {time}\n- History: {hist}\n\n"
                "Candidates:\n{cand}\n\n"
                "Task: Evaluate how much you want to visit each candidate (0-10).\n"
                "Format:\n"
                "[Name 1]: [Score]\n"
                "[Name 2]: [Score]\n"
                "...\n"
                "Reasoning: [One sentence summary]"
            )
        ])

    def decide(self, candidates):
        hist_str = " -> ".join(self.history) if self.history else "None"
        cand_str = "\n".join([f"- {c['name']} ({c['category']}, {c['dist']}m)" for c in candidates])
        
        msg = self.prompt_template.format_messages(
            persona=self.persona, loc=self.current_location_name,
            time=self.current_time.strftime("%H:%M"), hist=hist_str, cand=cand_str
        )
        response = self.llm.invoke(msg).content
        
        # --- スコア抽出と確率的選択ロジック ---
        scores = {}
        for line in response.split('\n'):
            # "高遠城址公園: 8" のような行を探す
            if ':' in line and not line.startswith("Reasoning"):
                parts = line.split(':')
                name = parts[0].strip().replace("- ", "")
                try:
                    score = float(parts[1].strip())
                    scores[name] = score
                except ValueError:
                    continue
        
        # 候補リストにあるものだけスコアを採用（パース失敗対策）
        valid_candidates = []
        weights = []
        
        for c in candidates:
            # LLMがスコアを付けなかった場合はデフォルト1点
            s = scores.get(c['name'], 1.0)
            valid_candidates.append(c['name'])
            # スコアを重みとして使う (2乗することで差を広げるテクニックも有効)
            weights.append(s ** 2) 

        if not valid_candidates:
            return response, "Error: No valid decision"

        # Pythonの機能で重み付け抽選を行う
        # k=1 で1つ選ぶ
        decision = random.choices(valid_candidates, weights=weights, k=1)[0]
        
        self.current_location_name = decision
        self.history.append(decision)
        
        return response, decision
