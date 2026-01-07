import numpy as np
from datetime import datetime
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

class ClusterVerbalizerEn:
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def verbalize(self, cluster_id, centroid_matrix):
        idx_time = [i for i, f in enumerate(self.feature_names) if 'start_hour' in f][0]
        avg_time = np.mean(centroid_matrix[:, idx_time]) * 24.0
        time_desc = "You prefer morning activities." if avg_time < 12 else "You prefer afternoon/evening activities."
        
        cat_indices = [i for i, f in enumerate(self.feature_names) if 'cat_' in f]
        if cat_indices:
            mean_cats = np.mean(centroid_matrix[:, cat_indices], axis=0)
            top_cat = self.feature_names[cat_indices[np.argmax(mean_cats)]].replace('cat_', '')
        else:
            top_cat = "exploring"
        
        return (f"You are a tourist visiting Ina City. ID: {cluster_id}.\n"
                f"Personality: {time_desc} You strongly prefer '{top_cat}'.\n"
                f"Guideline: Decide based on your personality.")

class TouristAgentEn:
    def __init__(self, persona_text, model_name="llama3"):
        self.persona = persona_text
        self.current_location_name = "Start Point"
        self.current_time = datetime(2023, 11, 1, 10, 0)
        self.history = []
        self.llm = ChatOllama(model=model_name, temperature=0.7)
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{persona}"),
            HumanMessagePromptTemplate.from_template(
                "Situation:\n- Location: {loc}\n- Time: {time}\n- History: {hist}\n\n"
                "Candidates:\n{cand}\n\n"
                "Task: Pick one destination from the Candidates.\n"
                "Format:\nDecision: [Name from list]\nReasoning: [English sentence]"
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
        
        decision = "Unknown"
        for line in response.split('\n'):
            if line.startswith("Decision:"):
                decision = line.replace("Decision:", "").strip()
                self.current_location_name = decision
                self.history.append(decision)
                break
        return response, decision