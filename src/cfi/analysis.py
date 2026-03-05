# import json
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats

# # ==========================================
# # CẤU HÌNH HIỂN THỊ BIỂU ĐỒ CHUẨN PAPER
# # ==========================================
# sns.set_theme(style="whitegrid", palette="muted")
# plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'savefig.dpi': 300})

# # ==========================================
# # 1. ĐỌC VÀ BÓC TÁCH DỮ LIỆU TỪ JSONL
# # ==========================================
# file_path = r'D:\Code\Python\Research\LLMs_reviewer\Benchmarking\output_cfi\cfi_results_cleaned.jsonl' # Thay bằng tên file thực tế của bạn

# rows_individual = []
# rows_paper = []
# topic_stats = []
# complementary_cases = []
# dealbreakers = {}

# try:
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if not line.strip(): continue
#             data = json.loads(line)
            
#             paper_id = data.get('paper_id', 'unknown')
#             scores = data.get('scores', {})
#             details = data.get('details', {})
#             flaw_weights = data.get('flaw_weights', {})
            
#             # --- XỬ LÝ ĐIỂM SỐ (RQ1, RQ3, RQ4) ---
#             llm_score = scores.get('LLM_Reviewer', np.nan)
#             meta_score = scores.get('Meta_Reviewer', np.nan)
            
#             human_scores = []
#             # Thêm điểm của từng Human vào danh sách
#             for k, v in scores.items():
#                 if 'Human' in k:
#                     human_scores.append(v)
#                     rows_individual.append({
#                         'paper_id': paper_id,
#                         'Reviewer_Role': 'Individual Human',
#                         'Score': v,
#                         'Meta_Score': meta_score
#                     })
            
#             # Thêm điểm của LLM vào danh sách
#             if not pd.isna(llm_score):
#                 rows_individual.append({
#                     'paper_id': paper_id,
#                     'Reviewer_Role': 'LLM Reviewer',
#                     'Score': llm_score,
#                     'Meta_Score': meta_score
#                 })
                
#             # --- TÍNH BEST HUMAN & THÊM VÀO DANH SÁCH ---
#             best_human = np.max(human_scores) if human_scores else np.nan
#             human_std = np.std(human_scores) if len(human_scores) > 1 else np.nan
            
#             # QUAN TRỌNG: Thêm 'Best Human' như một "nhân vật" riêng biệt để vẽ biểu đồ
#             if not pd.isna(best_human):
#                 rows_individual.append({
#                     'paper_id': paper_id,
#                     'Reviewer_Role': 'Best Human',
#                     'Score': best_human,
#                     'Meta_Score': meta_score
#                 })
            
#             rows_paper.append({
#                 'paper_id': paper_id,
#                 'LLM_Score': llm_score,
#                 'Best_Human_Score': best_human,
#                 'Meta_Score': meta_score,
#                 'Human_Std': human_std
#             })
            
#             # --- XỬ LÝ DETAILS VÀ FLAWS (RQ2, RQ5, RQ6) ---
#             for topic, reviewers_dict in details.items():
#                 # Bỏ qua topic không xác định
#                 if topic == "Other / Uncategorized": continue
                
#                 found_by_llm = 1 if len(reviewers_dict.get('LLM_Reviewer', [])) > 0 else 0
#                 found_by_meta = 1 if len(reviewers_dict.get('Meta_Reviewer', [])) > 0 else 0
                
#                 human_keys = [k for k in reviewers_dict.keys() if 'Human' in k]
#                 human_hits = sum(1 for k in human_keys if len(reviewers_dict[k]) > 0)
#                 total_humans = len(human_keys) if len(human_keys) > 0 else 1
                
#                 # RQ2 Data
#                 topic_stats.append({
#                     'Paper_ID': paper_id,
#                     'Topic': topic,
#                     'LLM_Hit': found_by_llm,
#                     'Human_Avg_Hit': human_hits / total_humans
#                 })
                
#                 # RQ5 Data: LLM tìm ra, Meta đồng ý, nhưng <= 1 Human tìm ra (Bị hội đồng bỏ sót)
#                 if found_by_llm == 1 and found_by_meta == 1 and human_hits <= 1 and total_humans >= 2:
#                     complementary_cases.append({'Paper_ID': paper_id, 'Topic': topic})
                    
#                 # RQ6 Data: Deal-breakers
#                 if topic not in dealbreakers:
#                     dealbreakers[topic] = 0
#                 dealbreakers[topic] += flaw_weights.get(topic, 0)

# except FileNotFoundError:
#     print(f"Không tìm thấy file {file_path}. Vui lòng kiểm tra lại đường dẫn.")
#     exit()

# # Tạo DataFrames
# df_indiv = pd.DataFrame(rows_individual).dropna(subset=['Score'])
# df_paper = pd.DataFrame(rows_paper).dropna(subset=['LLM_Score', 'Best_Human_Score'])
# df_topics = pd.DataFrame(topic_stats)
# df_comp = pd.DataFrame(complementary_cases)

# print(f"\n[DATA INFO] Đã xử lý {len(df_paper)} bài báo với {len(df_indiv)} lượt review.")

# # ==========================================
# # RQ1: OVERALL CAPABILITY (LLM VS HUMANS)
# # ==========================================
# print("\n" + "="*50)
# print("RQ1: OVERALL CAPABILITY ALIGNMENT")
# print("="*50)

# mean_llm = df_paper['LLM_Score'].mean()
# mean_best_human = df_paper['Best_Human_Score'].mean()
# mean_indiv_human = df_indiv[df_indiv['Reviewer_Role'] == 'Individual Human']['Score'].mean()

# print(f"- Điểm trung bình 1 Human ngẫu nhiên: {mean_indiv_human:.2f}")
# print(f"- Điểm trung bình của LLM: {mean_llm:.2f}")
# print(f"- Điểm trung bình của Best Human: {mean_best_human:.2f}")

# t_stat, p_val = stats.ttest_rel(df_paper['LLM_Score'], df_paper['Best_Human_Score'])
# print(f"- Paired T-test (LLM vs Best Human): t={t_stat:.3f}, p={p_val:.4e}")

# # Biểu đồ RQ1
# plt.figure(figsize=(9, 6))
# # Đảm bảo thứ tự hiển thị trên biểu đồ theo ý muốn (Individual -> LLM -> Best)
# order = ['Individual Human', 'LLM Reviewer', 'Best Human']

# sns.boxplot(x='Reviewer_Role', y='Score', hue='Reviewer_Role', data=df_indiv, 
#             order=order, width=0.5, palette="Set2", showfliers=False, legend=False)

# # (Tùy chọn) Thêm stripplot để biểu đồ sinh động hơn, nếu dữ liệu quá lớn (27k điểm) thì có thể bỏ qua hoặc dùng 'alpha' nhỏ
# # sns.stripplot(x='Reviewer_Role', y='Score', data=df_indiv, order=order, color=".3", alpha=0.05, jitter=True, size=2)

# plt.title("RQ1: Score Distribution (Individual Human vs. LLM vs. Best Human)", fontweight='bold')
# plt.ylabel("Consensus Score")
# plt.xlabel("")
# plt.tight_layout()
# plt.savefig("RQ1_Capability.png")
# plt.close()

# print("Đã vẽ xong biểu đồ RQ1_Capability.png")

# # ==========================================
# # RQ2: TOPIC-SPECIFIC PROFICIENCY
# # ==========================================
# print("\n" + "="*50)
# print("RQ2: TOPIC-SPECIFIC PROFICIENCY & BIAS")
# print("="*50)

# rq2_summary = df_topics.groupby('Topic').agg(
#     LLM_Hit_Rate=('LLM_Hit', lambda x: np.mean(x) * 100),
#     Human_Hit_Rate=('Human_Avg_Hit', lambda x: np.mean(x) * 100)
# ).reset_index()

# print(rq2_summary.to_string(index=False, float_format="%.1f%%"))

# # Biểu đồ RQ2
# df_melted_rq2 = rq2_summary.melt(id_vars='Topic', value_vars=['LLM_Hit_Rate', 'Human_Hit_Rate'], 
#                                  var_name='Reviewer', value_name='Hit Rate (%)')
# df_melted_rq2['Reviewer'] = df_melted_rq2['Reviewer'].replace({'LLM_Hit_Rate': 'LLM', 'Human_Hit_Rate': 'Human Average'})

# plt.figure(figsize=(12, 6))
# sns.barplot(data=df_melted_rq2, y='Topic', x='Hit Rate (%)', hue='Reviewer', palette=['#ff9999', '#66b3ff'])
# plt.title('RQ2: Flaw Detection Hit Rate by Topic', fontweight='bold')
# plt.ylabel("")
# plt.tight_layout()
# plt.savefig('RQ2_Topic_Proficiency.png')
# plt.close()

# # ==========================================
# # RQ3: ALIGNMENT WITH META-REVIEWER
# # ==========================================
# print("\n" + "="*50)
# print("RQ3: ALIGNMENT WITH META-REVIEWER")
# print("="*50)

# df_paper_meta = df_paper.dropna(subset=['LLM_Score', 'Meta_Score'])
# df_human_meta = df_indiv[(df_indiv['Reviewer_Role'] == 'Individual Human')].dropna(subset=['Score', 'Meta_Score'])

# if len(df_paper_meta) > 1:
#     corr_llm, _ = stats.pearsonr(df_paper_meta['LLM_Score'], df_paper_meta['Meta_Score'])
#     corr_hum, _ = stats.pearsonr(df_human_meta['Score'], df_human_meta['Meta_Score'])
#     print(f"- Tương quan (LLM & Meta): r = {corr_llm:.3f}")
#     print(f"- Tương quan (1 Human ngẫu nhiên & Meta): r = {corr_hum:.3f}")

#     # Biểu đồ RQ3
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     sns.regplot(x='Meta_Score', y='LLM_Score', data=df_paper_meta, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
#     plt.title(f"LLM vs Meta-Reviewer (r = {corr_llm:.2f})")
    
#     plt.subplot(1, 2, 2)
#     sns.regplot(x='Meta_Score', y='Score', data=df_human_meta, scatter_kws={'alpha':0.5, 'color':'green'}, line_kws={'color':'red'})
#     plt.title(f"Individual Human vs Meta-Reviewer (r = {corr_hum:.2f})")
#     plt.tight_layout()
#     plt.savefig("RQ3_Meta_Alignment.png")
#     plt.close()
# else:
#     print("Không đủ dữ liệu Meta_Score để tính toán RQ3.")

# # ==========================================
# # RQ4: HUMAN VARIANCE
# # ==========================================
# print("\n" + "="*50)
# print("RQ4: HUMAN REVIEWER VARIANCE (SUBJECTIVITY)")
# print("="*50)
# mean_human_std = df_paper['Human_Std'].mean()
# print(f"- Độ lệch chuẩn (Std) trung bình giữa các Human trên cùng 1 bài báo: {mean_human_std:.2f}")

# # ==========================================
# # RQ5: COMPLEMENTARY VALUE (BLIND SPOTS)
# # ==========================================
# print("\n" + "="*50)
# print("RQ5: COMPLEMENTARY VALUE (AI AS A COPILOT)")
# print("="*50)
# print(f"Tổng số trường hợp LLM tìm ra lỗi được Meta xác nhận nhưng Human bỏ sót: {len(df_comp)}")

# if len(df_comp) > 0:
#     comp_counts = df_comp['Topic'].value_counts().reset_index()
#     comp_counts.columns = ['Topic', 'Count']
#     print(comp_counts.to_string(index=False))

#     # Biểu đồ RQ5
#     plt.figure(figsize=(9, 6))
#     sns.barplot(data=comp_counts, y='Topic', x='Count', hue='Topic', palette="viridis", legend=False)
#     plt.title("RQ5: Blind Spots Recovered by LLM", fontweight='bold')
#     plt.ylabel("")
#     plt.xlabel("Number of Cases")
#     plt.tight_layout()
#     plt.savefig('RQ5_Complementary_Value.png')
#     plt.close()

# # ==========================================
# # RQ6: DEAL-BREAKERS (FLAW WEIGHTS)
# # ==========================================
# print("\n" + "="*50)
# print("RQ6: DEAL-BREAKER TOPICS (QUALITY IMPACT)")
# print("="*50)

# total_weight = sum(dealbreakers.values())
# rq6_list = [{'Topic': k, 'Percentage (%)': (v / total_weight) * 100} for k, v in dealbreakers.items()]
# df_rq6 = pd.DataFrame(rq6_list).sort_values('Percentage (%)', ascending=False)

# print(df_rq6.to_string(index=False, float_format="%.1f%%"))

# # Biểu đồ RQ6
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Percentage (%)', y='Topic', hue='Topic', data=df_rq6, palette="rocket", legend=False)
# plt.title("RQ6: Most Critical Flaws (Proportion of Consensus Weight)", fontweight='bold')
# plt.ylabel("")
# for index, value in enumerate(df_rq6['Percentage (%)']):
#     plt.text(value + 0.5, index, f"{value:.1f}%", va='center')
# plt.tight_layout()
# plt.savefig('RQ6_Deal_Breakers.png')
# plt.close()

# print("\n[HOÀN TẤT] Phân tích thành công. Toàn bộ biểu đồ đã được lưu.")

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==========================================
# CẤU HÌNH HIỂN THỊ BIỂU ĐỒ CHUẨN PAPER
# ==========================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'savefig.dpi': 300})


def get_semantic_micro_label(raw_label, macro_category):
    lbl = raw_label.lower()
    
    if macro_category == "Clarity & Presentation":
        if any(w in lbl for w in ["figure", "table", "visual", "caption", "illustration"]): return "Poor Figures/Tables Quality"
        if any(w in lbl for w in ["typo", "grammar", "english", "spelling"]): return "Grammar & Typos"
        if any(w in lbl for w in ["notation", "equation", "math"]): return "Unclear Math/Notations"
        if any(w in lbl for w in ["structure", "organization", "flow"]): return "Poor Paper Structure"
        return "General Writing & Clarity Issues" 
        
    if macro_category == "Experimental Design & Evaluation":
        if any(w in lbl for w in ["baseline", "state-of-the-art", "sota", "comparison"]): return "Missing/Weak Baselines"
        if any(w in lbl for w in ["ablation"]): return "Lack of Ablation Studies"
        if any(w in lbl for w in ["metric", "evaluation"]): return "Questionable Evaluation Metrics"
        if any(w in lbl for w in ["dataset", "data", "sample"]): return "Limited/Biased Datasets"
        if "hyperparameter" in lbl: return "Missing Hyperparameter Details"
        return "Insufficient Experimental Validation (General)"
        
    if macro_category == "Applicability, Scalability & Limitations":
        if "limitations" in lbl: return "Lack of Discussion on Limitations"
        if any(w in lbl for w in ["scalability", "computational", "complexity", "overhead", "time", "memory", "cost"]): return "Scalability & Complexity Concerns"
        if any(w in lbl for w in ["societal", "ethical", "broader", "privacy"]): return "Missing Broader Impact/Ethical Concerns"
        if any(w in lbl for w in ["generalization", "ood", "out-of-distribution"]): return "Poor Generalization/Robustness"
        return "General Applicability Issues"
        
    if macro_category == "Novelty & Contribution":
        if "incremental" in lbl: return "Incremental Contribution Only"
        if "significance" in lbl: return "Lack of Significance/Impact"
        return "Limited Novelty (General)"
        
    if macro_category == "Related Work & Citations":
        if "comparison" in lbl: return "Missing Comparisons with Prior Work"
        if any(w in lbl for w in ["concurrent", "recent"]): return "Missing Recent/Concurrent Works"
        return "Missing Relevant Citations (General)"
        
    if macro_category == "Methodology & Theoretical Soundness":
        if any(w in lbl for w in ["theory", "theoretical", "proof", "bound", "guarantee"]): return "Weak Theoretical Justification/Proofs"
        if "assumption" in lbl: return "Strong/Unrealistic Assumptions"
        if any(w in lbl for w in ["intuition", "justification"]): return "Lack of Intuition/Justification"
        return "Methodological Flaws (General)"
        
    if macro_category == "Reproducibility & Open Science":
        if any(w in lbl for w in ["code", "open source", "repository"]): return "Missing Code/Data Repository"
        if "detail" in lbl: return "Insufficient Implementation Details"
        return "General Reproducibility Concerns"

    return raw_label.title()

# ==========================================
# HÀM PHÂN LOẠI (MAPPING) TỪ MICRO SANG MACRO
# ==========================================
def categorize_label(raw_label):
    label = str(raw_label).lower().strip()
    prefixes = ["lack of ", "limited ", "insufficient ", "incomplete ", "inadequate ", 
                "missing ", "marginal ", "misleading ", "need for ", "questionable ", 
                "questions about ", "questions on ", "potential for ", "potential ", 
                "question ", "unclear ", "unaddressed ", "unanswered ", "unconvincing ", 
                "unjustified ", "unsupported ", "unfair ", "unrealistic ", "weak ", "under "]
    for prefix in prefixes:
        if label.startswith(prefix):
            label = label.replace(prefix, "", 1).strip()
            
    if any(k in label for k in ["ambiguit", "caption", "clarif", "clarity", "confusing", "difficulty", "editorial", "english", "equation", "error", "explanation", "exposition", "figure", "fig ", "formatting", "grammar", "illustration", "inaccurate", "incorrect", "misleading", "inconsistent", "intuition", "notation", "organization", "overclaim", "overstate", "presentation", "readability", "structure", "table", "terminology", "typo", "visual", "wording", "writing"]) or ("detail" in label and "implementation" not in label): return "Clarity & Presentation"
    if any(k in label for k in ["anonymity", "availability", "code", "empty repository", "implementation", "open source", "release", "reproducibility"]): return "Reproducibility & Open Science"
    if any(k in label for k in ["citation", "concurrent work", "contextualization", "exclusion of", "existing", "literature", "omission of", "outdated", "prior work", "prior art", "reference", "related work"]): return "Related Work & Citations"
    if any(k in label for k in ["advantage", "benefit", "contribution", "incremental", "innovation", "marginal", "motivation", "novelty", "obvious", "originality", "significance", "trivial", "uniqueness"]): return "Novelty & Contribution"
    if any(k in label for k in ["applicability", "broader impact", "complexity", "computational", "constraint", "cost", "discussion", "efficiency", "ethical", "extension", "flops", "future work", "generaliz", "gpu", "hardware", "hallucination", "handling", "latency", "limitation", "memory", "ood", "out-of-distribution", "overhead", "overfitting", "underfitting", "practical", "privacy", "real-world", "robustness", "runtime", "scalab", "scale", "scope", "societal", "speedup", "time"]): return "Applicability, Scalability & Limitations"
    if any(k in label for k in ["ablation", "accuracy", "analysis", "baseline", "benchmark", "comparison", "effectiveness", "empirical", "evaluation", "experiment", "fairness", "hyperparameter", "metric", "performance", "qualitative", "quantitative", "result", "sample", "validation", "zero-shot", "few-shot"]) or ("data" in label and "availability" not in label) or ("dataset" in label): return "Experimental Design & Evaluation"
    if any(k in label for k in ["algorithm", "architecture", "assumption", "backbone", "bound", "choice of", "convergence", "derivation", "equivalence", "expressiv", "formulation", "gap between", "guarantee", "heuristic", "hypothesis", "identifiability", "justification", "loss function", "method", "optimization", "parameter", "proof", "role of", "theoretical", "theory", "topology", "variance"]): return "Methodology & Theoretical Soundness"
    
    return "Other / Uncategorized"

# ==========================================
# 1. ĐỌC VÀ BÓC TÁCH DỮ LIỆU TỪ JSONL
# ==========================================
# TRỎ VÀO FILE CHỨA NHÃN GỐC (DETAILED) CỦA BẠN
file_path = r'D:\Code\Python\Research\LLMs_reviewer\Benchmarking\output_cfi\cfi_results_detailed.jsonl'

rows_individual = []
rows_paper = []
topic_stats = []
complementary_cases_micro = [] # Lưu nhãn gốc cho điểm mù
dealbreakers_macro = {}        # Tổng hợp trọng số theo 7 nhóm lớn
dealbreakers_micro = {}        # Tổng hợp trọng số theo nhãn gốc chi tiết

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            paper_id = data.get('paper_id', 'unknown')
            scores = data.get('scores', {})
            details = data.get('details', {})
            flaw_weights = data.get('flaw_weights', {})
            
            # --- XỬ LÝ ĐIỂM SỐ CÁ NHÂN (RQ1, RQ3, RQ4) ---
            llm_score = scores.get('LLM_Reviewer', np.nan)
            meta_score = scores.get('Meta_Reviewer', np.nan)
            human_scores = []
            
            for k, v in scores.items():
                if 'Human' in k:
                    human_scores.append(v)
                    rows_individual.append({'paper_id': paper_id, 'Reviewer_Role': 'Individual Human', 'Score': v, 'Meta_Score': meta_score})
            
            if not pd.isna(llm_score):
                rows_individual.append({'paper_id': paper_id, 'Reviewer_Role': 'LLM Reviewer', 'Score': llm_score, 'Meta_Score': meta_score})
                
            best_human = np.max(human_scores) if human_scores else np.nan
            human_std = np.std(human_scores) if len(human_scores) > 1 else np.nan
            
            if not pd.isna(best_human):
                rows_individual.append({'paper_id': paper_id, 'Reviewer_Role': 'Best Human', 'Score': best_human, 'Meta_Score': meta_score})
            
            rows_paper.append({'paper_id': paper_id, 'LLM_Score': llm_score, 'Best_Human_Score': best_human, 'Meta_Score': meta_score, 'Human_Std': human_std})
            
            # --- XỬ LÝ DETAILS (MACRO & MICRO) CHO RQ2, RQ5, RQ6 ---
            total_humans_in_paper = len([k for k in scores.keys() if 'Human' in k])
            paper_macro_topics = {} # Lưu trạng thái (ai tìm ra) gộp theo 7 nhóm cho RQ2
            
            for raw_flaw, reviewers_dict in details.items():
                macro_topic = categorize_label(raw_flaw)
                if macro_topic == "Other / Uncategorized": continue
                
                # Trạng thái của nhãn gốc (Micro)
                found_by_llm = 1 if len(reviewers_dict.get('LLM_Reviewer', [])) > 0 else 0
                found_by_meta = 1 if len(reviewers_dict.get('Meta_Reviewer', [])) > 0 else 0
                human_keys = [k for k in reviewers_dict.keys() if 'Human' in k]
                human_hits_micro = sum(1 for k in human_keys if len(reviewers_dict[k]) > 0)
                
                # --- GOM NHÓM LÊN MACRO (Cho RQ2) ---
                if macro_topic not in paper_macro_topics:
                    paper_macro_topics[macro_topic] = {'LLM': False, 'Humans': set()}
                if found_by_llm: paper_macro_topics[macro_topic]['LLM'] = True
                for hk in human_keys:
                    if len(reviewers_dict[hk]) > 0: paper_macro_topics[macro_topic]['Humans'].add(hk)
                
                # --- LƯU TRỮ MICRO (Cho RQ5 - Blind spots) ---
                # Tính điểm mù trên nhãn GỐC (Nếu LLM thấy, Meta thấy, <=1 Human thấy)
                if found_by_llm == 1 and found_by_meta == 1 and human_hits_micro <= 1 and total_humans_in_paper >= 2:
                    complementary_cases_micro.append({
                        'Paper_ID': paper_id, 
                        'Macro_Topic': macro_topic, 
                        'Specific_Flaw': raw_flaw,
                        'Human_Hits': human_hits_micro
                    })
                
                # --- LƯU TRỮ TRỌNG SỐ (Cho RQ6 - Dealbreakers) ---
                weight = flaw_weights.get(raw_flaw, 0)
                dealbreakers_macro[macro_topic] = dealbreakers_macro.get(macro_topic, 0) + weight
                dealbreakers_micro[raw_flaw] = dealbreakers_micro.get(raw_flaw, 0) + weight

            # Cập nhật RQ2 (Hit Rate) dựa trên Macro
            for m_topic, status in paper_macro_topics.items():
                topic_stats.append({
                    'Paper_ID': paper_id,
                    'Topic': m_topic,
                    'LLM_Hit': 1 if status['LLM'] else 0,
                    'Human_Avg_Hit': len(status['Humans']) / total_humans_in_paper if total_humans_in_paper > 0 else 0
                })

except FileNotFoundError:
    print(f"Không tìm thấy file {file_path}.")
    exit()

# Tạo DataFrames

df_indiv = pd.DataFrame(rows_individual).dropna(subset=['Score'])
df_paper = pd.DataFrame(rows_paper).dropna(subset=['LLM_Score', 'Best_Human_Score'])
df_topics = pd.DataFrame(topic_stats)
df_comp_micro = pd.DataFrame(complementary_cases_micro)

# (1) Tính mean điểm Individual Human theo paper (paper-level)
df_indiv_only = df_indiv[df_indiv['Reviewer_Role'] == 'Individual Human'].copy()
df_indiv_paper_mean = (
    df_indiv_only
    .groupby('paper_id', as_index=False)['Score']
    .mean()
    .rename(columns={'Score': 'IndivHuman_MeanScore'})
)

print(f"\n[DATA INFO] Đã xử lý {len(df_paper)} bài báo với {len(df_indiv)} lượt review.")

# ==========================================
# RQ1: OVERALL CAPABILITY (LLM VS HUMANS)
# ==========================================
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

print("\n" + "="*50)
print("RQ1: OVERALL CAPABILITY ALIGNMENT")
print("="*50)

# (2) Tính các mean ở PAPER-LEVEL cho cả 3 nhóm
mean_llm_paper        = df_paper['LLM_Score'].mean()
mean_best_human_paper = df_paper['Best_Human_Score'].mean()

# Lưu ý: để so mean paper-level của Individual Human với cùng tập paper như df_paper,
# ta merge để chỉ lấy các paper có cả LLM_Score/Best_Human_Score và có IndivHuman_MeanScore.
df_plot = df_paper.merge(df_indiv_paper_mean, on='paper_id', how='inner')

mean_indiv_human_paper = df_plot['IndivHuman_MeanScore'].mean()

print(f"- Individual Human (paper-mean): {mean_indiv_human_paper:.2f} | "
      f"LLM: {mean_llm_paper:.2f} | Best Human: {mean_best_human_paper:.2f}")

# (3) Paired T-test (LLM vs Best Human) trên paper-level (giữ nguyên, nhưng chắc chắn dropna)
df_pair_bh = df_paper.dropna(subset=['LLM_Score', 'Best_Human_Score'])
if len(df_pair_bh) >= 2:
    t_stat_bh, p_val_bh = stats.ttest_rel(df_pair_bh['LLM_Score'], df_pair_bh['Best_Human_Score'])
    print(f"- Paired T-test (LLM vs Best Human): t={t_stat_bh:.3f}, p={p_val_bh:.4e} (n={len(df_pair_bh)})")
else:
    print("- Paired T-test (LLM vs Best Human): KHÔNG ĐỦ CẶP (n<2)")

# (4) NEW: Paired T-test (LLM vs Individual Human - MEAN per paper)
# Dùng df_plot (đã inner-join) để đảm bảo cùng tập paper cho 2 đại lượng.
df_pair_indiv = df_plot.dropna(subset=['LLM_Score', 'IndivHuman_MeanScore'])
if len(df_pair_indiv) >= 2:
    t_stat_indiv, p_val_indiv = stats.ttest_rel(df_pair_indiv['LLM_Score'], df_pair_indiv['IndivHuman_MeanScore'])
    print(f"- Paired T-test (LLM vs Individual Human - MEAN per paper): t={t_stat_indiv:.3f}, p={p_val_indiv:.4e} (n={len(df_pair_indiv)})")
else:
    print("- Paired T-test (LLM vs Individual Human - MEAN per paper): KHÔNG ĐỦ CẶP (n<2)")

# (5) Chuẩn hóa dữ liệu vẽ BOX PLOT về PAPER-LEVEL cho cả 3 nhóm
plot_rows = []
for _, r in df_plot.iterrows():
    if not pd.isna(r['IndivHuman_MeanScore']):
        plot_rows.append({'Reviewer_Role': 'Individual Human (paper-mean)', 'Score': r['IndivHuman_MeanScore']})
    if not pd.isna(r['LLM_Score']):
        plot_rows.append({'Reviewer_Role': 'LLM Reviewer', 'Score': r['LLM_Score']})
    if not pd.isna(r['Best_Human_Score']):
        plot_rows.append({'Reviewer_Role': 'Best Human', 'Score': r['Best_Human_Score']})

df_plot_long = pd.DataFrame(plot_rows)

# (6) Vẽ boxplot PAPER-LEVEL (Phương án A)
plt.figure(figsize=(9, 6))
order = ['Individual Human (paper-mean)', 'LLM Reviewer', 'Best Human']
sns.boxplot(
    x='Reviewer_Role', y='Score', data=df_plot_long,
    order=order, width=0.5, palette="Set2", showfliers=False
)
plt.title("Paper-level Score Distribution: Indiv Human (mean) vs LLM vs Best Human", fontweight='bold')
plt.ylabel("Consensus Score")
plt.xlabel("")
plt.tight_layout()
plt.savefig("RQ1_Capability_paper_level.png")
plt.close()

df_indiv_only = df_indiv[df_indiv['Reviewer_Role'] == 'Individual Human'].copy()

# 2) Tính các chỉ số phân tán theo từng paper
def iqr(x):
    # Inter-Quartile Range (Q3 - Q1)
    return np.nanquantile(x, 0.75) - np.nanquantile(x, 0.25)

def mad(x):
    # Median Absolute Deviation (robust)
    # Nếu scipy phiên bản mới có stats.median_abs_deviation, bạn có thể dùng thay
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))

agg = df_indiv_only.groupby('paper_id').agg(
    Human_Mean=('Score', 'mean'),
    Human_Std_sample=('Score', lambda x: np.std(x, ddof=1) if len(x) > 1 else np.nan),
    Human_IQR=('Score', iqr),
    Human_Range=('Score', lambda x: np.nanmax(x) - np.nanmin(x) if len(x) > 0 else np.nan),
    Human_MAD=('Score', mad),
    Human_Max=('Score', 'max'),
    Human_Min=('Score', 'min'),
    n_reviewers=('Score', 'size')
).reset_index()

# 3) Ghép với Best_Human_Score và LLM_Score để tính các chênh lệch hữu ích
df_dispersion = (agg
    .merge(df_paper[['paper_id', 'Best_Human_Score', 'LLM_Score']], on='paper_id', how='left')
    .merge(df_indiv_paper_mean, on='paper_id', how='left')  # đã có IndivHuman_MeanScore từ RQ1-A
)

# 4) Các khoảng cách quan trọng
df_dispersion['BestMinusMean'] = df_dispersion['Best_Human_Score'] - df_dispersion['IndivHuman_MeanScore']
df_dispersion['BestMinusWorst'] = df_dispersion['Best_Human_Score'] - df_dispersion['Human_Min']  # biên trên-dưới
df_dispersion['LLMMinusMean']  = df_dispersion['LLM_Score'] - df_dispersion['IndivHuman_MeanScore']

# 5) Tóm tắt toàn tập (distribution summary)
def summarize(series):
    s = series.dropna().to_numpy()
    if s.size == 0:
        return {}
    return {
        'count': s.size,
        'mean': np.mean(s),
        'std': np.std(s, ddof=1) if s.size > 1 else np.nan,
        'median': np.median(s),
        'p10': np.quantile(s, 0.10),
        'p25': np.quantile(s, 0.25),
        'p75': np.quantile(s, 0.75),
        'p90': np.quantile(s, 0.90),
        'max': np.max(s)
    }

summary_hstd = summarize(df_dispersion['Human_Std_sample'])
summary_bm   = summarize(df_dispersion['BestMinusMean'])
summary_range= summarize(df_dispersion['Human_Range'])

print("\n[HUMAN DISAGREEMENT — TỔNG QUAN]")
print(f"- Human Std (sample) — median: {summary_hstd.get('median', np.nan):.2f}, "
      f"IQR: [{summary_hstd.get('p25', np.nan):.2f}, {summary_hstd.get('p75', np.nan):.2f}], "
      f"p90: {summary_hstd.get('p90', np.nan):.2f}")
print(f"- Best − Mean — median: {summary_bm.get('median', np.nan):.2f}, "
      f"IQR: [{summary_bm.get('p25', np.nan):.2f}, {summary_bm.get('p75', np.nan):.2f}], "
      f"p90: {summary_bm.get('p90', np.nan):.2f}")
print(f"- Human Range — median: {summary_range.get('median', np.nan):.2f}")

# 6) Ngưỡng “disagreement cao” (dựa trên phân vị)
thr_std_p75 = df_dispersion['Human_Std_sample'].quantile(0.75)
thr_bm_p75  = df_dispersion['BestMinusMean'].quantile(0.75)
high_std_rate = np.mean(df_dispersion['Human_Std_sample'] > thr_std_p75) * 100
high_bm_rate  = np.mean(df_dispersion['BestMinusMean']  > thr_bm_p75)  * 100

print(f"- Tỷ lệ paper có Human Std vượt P75: {high_std_rate:.1f}% "
      f"(ngưỡng ~{thr_std_p75:.2f})")
print(f"- Tỷ lệ paper có Best−Mean vượt P75: {high_bm_rate:.1f}% "
      f"(ngưỡng ~{thr_bm_p75:.2f})")


# --- Guard: ensure needed cols exist ---
_required_cols = ['Human_Std_sample','BestMinusMean','Human_Range','IndivHuman_MeanScore','LLM_Score','n_reviewers']
missing = [c for c in _required_cols if c not in df_dispersion.columns]
if missing:
    print(f"[WARN] Thiếu cột trong df_dispersion: {missing}. Một số biểu đồ có thể bị bỏ qua.")

# ---------- 1) HISTOGRAM + KDE ----------
plt.figure(figsize=(8,5))
sns.histplot(df_dispersion['Human_Std_sample'].dropna(), bins=40, kde=True, color='#66b3ff')
q25, q50, q75 = np.nanquantile(df_dispersion['Human_Std_sample'], [0.25, 0.5, 0.75])
for q, lab, col in [(q25,'P25','#888'),(q50,'Median','#d62728'),(q75,'P75','#2ca02c')]:
    plt.axvline(q, ls='--', c=col, lw=1.5, label=f'{lab}: {q:.2f}')
plt.title('Phân phối Human Std (sample) giữa các reviewer')
plt.xlabel('Human Std (sample)')
plt.ylabel('Số paper')
plt.legend()
plt.tight_layout()
plt.savefig('viz_human_std_hist.png')
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(df_dispersion['BestMinusMean'].dropna(), bins=40, kde=True, color='#ff9999')
q25, q50, q75 = np.nanquantile(df_dispersion['BestMinusMean'], [0.25, 0.5, 0.75])
for q, lab, col in [(q25,'P25','#888'),(q50,'Median','#d62728'),(q75,'P75','#2ca02c')]:
    plt.axvline(q, ls='--', c=col, lw=1.5, label=f'{lab}: {q:.2f}')
plt.title('Phân phối khoảng cách Best − Mean (theo paper)')
plt.xlabel('Best − Mean')
plt.ylabel('Số paper')
plt.legend()
plt.tight_layout()
plt.savefig('viz_best_minus_mean_hist.png')
plt.close()

# ---------- 2) ECDF (để đọc phần trăm papers trên ngưỡng) ----------
from statsmodels.distributions.empirical_distribution import ECDF

plt.figure(figsize=(8,5))
vals = df_dispersion['Human_Std_sample'].dropna().to_numpy()
if vals.size:
    ec = ECDF(vals)
    x = np.linspace(vals.min(), vals.max(), 400)
    y = ec(x)
    plt.plot(x, y, label='ECDF Human Std', color='#1f77b4')
    thr = np.nanquantile(vals, 0.75)
    plt.axvline(thr, ls='--', c='#2ca02c', label=f'P75 ≈ {thr:.2f}')
    plt.axhline(0.75, ls=':', c='#2ca02c')
    plt.title('ECDF của Human Std (sample)')
    plt.xlabel('Human Std (sample)')
    plt.ylabel('F(x)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('viz_human_std_ecdf.png')
    plt.close()

# ---------- 3) BOX / VIOLIN cho 3 chỉ số chính ----------
_long_rows = []
for col, nice in [('Human_Std_sample','Human Std (sample)'),
                  ('BestMinusMean','Best − Mean'),
                  ('Human_Range','Human Range')]:
    if col in df_dispersion.columns:
        for v in df_dispersion[col].dropna().to_numpy():
            _long_rows.append({'Metric': nice, 'Value': v})

if _long_rows:
    df_long = pd.DataFrame(_long_rows)
    plt.figure(figsize=(9,5))
    # Seaborn >=0.14: dùng màu mặc định (không chỉ định palette) để tránh warning khi không có hue
    sns.violinplot(x='Metric', y='Value', data=df_long, inner='quartile', cut=0)
    plt.title('Phân phối các chỉ số bất đồng giữa reviewer con người')
    plt.xlabel('')
    plt.ylabel('Giá trị')
    plt.tight_layout()
    plt.savefig('viz_disagreement_violin.png')
    plt.close()

# ---------- 4) SCATTER: LLM vs Human (paper-mean) tô màu theo Human Std ----------
if set(['IndivHuman_MeanScore','LLM_Score','Human_Std_sample']).issubset(df_dispersion.columns):
    plt.figure(figsize=(6,6))
    sc = plt.scatter(
        df_dispersion['IndivHuman_MeanScore'],
        df_dispersion['LLM_Score'],
        c=df_dispersion['Human_Std_sample'], cmap='viridis', s=8, alpha=0.7
    )
    lim_min = np.nanmin([df_dispersion['IndivHuman_MeanScore'].min(), df_dispersion['LLM_Score'].min()])
    lim_max = np.nanmax([df_dispersion['IndivHuman_MeanScore'].max(), df_dispersion['LLM_Score'].max()])
    plt.plot([lim_min, lim_max],[lim_min, lim_max], ls='--', c='gray', label='y = x')
    plt.colorbar(sc, label='Human Std (sample)')
    plt.title('LLM vs Individual Human (paper-mean), tô màu theo Human Std')
    plt.xlabel('Individual Human (paper-mean)')
    plt.ylabel('LLM Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('viz_llm_vs_humanmean_colored_by_hstd.png')
    plt.close()

# ---------- 5) DISAGREEMENT vs số reviewer ----------
if set(['n_reviewers','Human_Std_sample']).issubset(df_dispersion.columns):
    plt.figure(figsize=(7,5))
    sns.boxplot(
        x=pd.cut(df_dispersion['n_reviewers'], bins=[0,2,3,4,6,10,100], include_lowest=True),
        y=df_dispersion['Human_Std_sample'],
        showfliers=False
    )
    plt.title('Mức bất đồng theo số lượng reviewer mỗi paper')
    plt.xlabel('Số reviewer (bins)')
    plt.ylabel('Human Std (sample)')
    plt.tight_layout()
    plt.savefig('viz_disagreement_by_reviewer_count.png')
    plt.close()

# ---------- 6) THEO TOPIC (nếu df_topics có) ----------
if 'df_topics' in globals() and {'paper_id','Topic'}.issubset(df_topics.columns):
    df_disp_topic = df_dispersion.merge(df_topics[['paper_id','Topic']], on='paper_id', how='left')
    # Sắp theo median để đọc dễ hơn
    order = (df_disp_topic.groupby('Topic')['Human_Std_sample']
             .median().sort_values(ascending=False).index.tolist())
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df_disp_topic, x='Topic', y='Human_Std_sample', order=order, showfliers=False)
    plt.title('Human Std (sample) theo Topic')
    plt.xlabel('')
    plt.ylabel('Human Std (sample)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('viz_human_std_by_topic.png')
    plt.close()

    # Top topic theo Best − Mean (median)
    order_bm = (df_disp_topic.groupby('Topic')['BestMinusMean']
                .median().sort_values(ascending=False).index.tolist())
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df_disp_topic, x='Topic', y='BestMinusMean', order=order_bm, showfliers=False)
    plt.title('Best − Mean theo Topic')
    plt.xlabel('')
    plt.ylabel('Best − Mean')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('viz_best_minus_mean_by_topic.png')
    plt.close()

print('[OK] Đã xuất các hình trực quan:')
print('- viz_human_std_hist.png')
print('- viz_best_minus_mean_hist.png')
print('- viz_human_std_ecdf.png (nếu đủ dữ liệu)')
print('- viz_disagreement_violin.png')
print('- viz_llm_vs_humanmean_colored_by_hstd.png')
print('- viz_disagreement_by_reviewer_count.png')
print('- viz_human_std_by_topic.png / viz_best_minus_mean_by_topic.png (nếu có Topic)')


# ==========================================
# RQ2: TOPIC-SPECIFIC PROFICIENCY
# ==========================================
print("\n" + "="*50)
print("RQ2: TOPIC-SPECIFIC PROFICIENCY & BIAS (MACRO LEVEL)")
print("="*50)
rq2_summary = df_topics.groupby('Topic').agg(
    LLM_Hit_Rate=('LLM_Hit', lambda x: np.mean(x) * 100),
    Human_Hit_Rate=('Human_Avg_Hit', lambda x: np.mean(x) * 100)
).reset_index()
print(rq2_summary.to_string(index=False, float_format="%.1f%%"))

# Biểu đồ RQ2
df_melted_rq2 = rq2_summary.melt(id_vars='Topic', value_vars=['LLM_Hit_Rate', 'Human_Hit_Rate'], var_name='Reviewer', value_name='Hit Rate (%)')
df_melted_rq2['Reviewer'] = df_melted_rq2['Reviewer'].replace({'LLM_Hit_Rate': 'LLM', 'Human_Hit_Rate': 'Human Average'})
plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted_rq2, y='Topic', x='Hit Rate (%)', hue='Reviewer', palette=['#ff9999', '#66b3ff'])
plt.title('Flaw Detection Hit Rate by Macro Topic', fontweight='bold')
plt.ylabel("")
plt.tight_layout()
plt.savefig('RQ2_Topic_Proficiency.png')
plt.close()

# ==========================================
# RQ3: ALIGNMENT WITH META-REVIEWER
# ==========================================
print("\n" + "="*50)
print("RQ3: ALIGNMENT WITH META-REVIEWER (STATISTICALLY RIGOROUS)")
print("="*50)

# 1. Lấy dữ liệu điểm từ df_paper, đảm bảo không có giá trị NaN ở cả 3 mảng
df_rq3 = df_paper.dropna(subset=['LLM_Score', 'Meta_Score', 'Best_Human_Score']).copy()

# 2. Tính điểm đại diện của Human cho TỪNG bài báo (Mean Human Score)
df_human_mean_rq3 = df_indiv[df_indiv['Reviewer_Role'] == 'Individual Human'].groupby('paper_id')['Score'].mean().reset_index()
df_human_mean_rq3.rename(columns={'Score': 'Human_Mean_Score'}, inplace=True)

# 3. Merge lại để đảm bảo mỗi row là 1 bài báo duy nhất
df_rq3 = df_rq3.merge(df_human_mean_rq3, on='paper_id', how='inner')

if len(df_rq3) > 1:
    # 4. Tính Pearson Correlation trên các quan sát độc lập
    corr_llm, p_llm = stats.pearsonr(df_rq3['LLM_Score'], df_rq3['Meta_Score'])
    corr_hum, p_hum = stats.pearsonr(df_rq3['Human_Mean_Score'], df_rq3['Meta_Score'])
    corr_best, p_best = stats.pearsonr(df_rq3['Best_Human_Score'], df_rq3['Meta_Score']) # TÍNH CHO BEST HUMAN
    
    print(f"- Số lượng quan sát độc lập (N): {len(df_rq3)} bài báo")
    print(f"- Tương quan (LLM & Meta)          : r = {corr_llm:.3f} (p-value = {p_llm:.2e})")
    print(f"- Tương quan (Average Human & Meta): r = {corr_hum:.3f} (p-value = {p_hum:.2e})")
    print(f"- Tương quan (Best Human & Meta)   : r = {corr_best:.3f} (p-value = {p_best:.2e})")

    # 5. Biểu đồ RQ3 (Cập nhật thành 3 cột - 1x3 grid)
    plt.figure(figsize=(16, 5)) # Kéo dài chiều ngang biểu đồ
    
    # Biểu đồ 1: LLM
    plt.subplot(1, 3, 1)
    sns.regplot(x='Meta_Score', y='LLM_Score', data=df_rq3, scatter_kws={'alpha':0.3, 'color':'#1f77b4'}, line_kws={'color':'red'})
    plt.title(f"LLM vs Meta-Reviewer\n(r = {corr_llm:.3f})", fontweight='bold')
    plt.xlabel("Meta-Reviewer Score")
    plt.ylabel("LLM Score")
    
    # Biểu đồ 2: Average Human
    plt.subplot(1, 3, 2)
    sns.regplot(x='Meta_Score', y='Human_Mean_Score', data=df_rq3, scatter_kws={'alpha':0.3, 'color':'#2ca02c'}, line_kws={'color':'red'})
    plt.title(f"Average Human vs Meta-Reviewer\n(r = {corr_hum:.3f})", fontweight='bold')
    plt.xlabel("Meta-Reviewer Score")
    plt.ylabel("Mean Human Score")
    
    # Biểu đồ 3: Best Human
    plt.subplot(1, 3, 3)
    sns.regplot(x='Meta_Score', y='Best_Human_Score', data=df_rq3, scatter_kws={'alpha':0.3, 'color':'#9467bd'}, line_kws={'color':'red'})
    plt.title(f"Best Human vs Meta-Reviewer\n(r = {corr_best:.3f})", fontweight='bold')
    plt.xlabel("Meta-Reviewer Score")
    plt.ylabel("Best Human Score")
    
    plt.tight_layout()
    plt.savefig("RQ3_Meta_Alignment.png")
    plt.close()
else:
    print("Không đủ dữ liệu Paper-level để tính toán RQ3.")
# ==========================================
# RQ5: COMPLEMENTARY VALUE (ĐIỂM MÙ TẬP THỂ)
# ==========================================
print("\n" + "="*50)
print("RQ5: COMPLEMENTARY VALUE (AI AS A COPILOT)")
print("="*50)
print(f"Tổng số trường hợp LLM 'cứu thua' cho hội đồng: {len(df_comp_micro)}")

if len(df_comp_micro) > 0:
    # Top Nhóm lớn chứa điểm mù
    macro_blindspots = df_comp_micro['Macro_Topic'].value_counts().reset_index()
    macro_blindspots.columns = ['Macro Topic', 'Count']
    print("\n--- Nhóm lỗi (Macro) mà con người hay bỏ sót nhất ---")
    print(macro_blindspots.to_string(index=False))
    
    # ---------------------------------------------------------
    # BƯỚC MỚI: GỘP NGỮ NGHĨA (SEMANTIC MERGING) CHO RQ5
    # ---------------------------------------------------------
    # Áp dụng hàm get_semantic_micro_label cho từng dòng trong DataFrame
    df_comp_micro['Semantic_Flaw'] = df_comp_micro.apply(
        lambda row: get_semantic_micro_label(row['Specific_Flaw'], row['Macro_Topic']), axis=1
    )
    
    # CHI TIẾT MICRO: Top 5 nhãn ĐÃ GỘP NGỮ NGHĨA bị bỏ sót nhiều nhất
    semantic_micro_blindspots = df_comp_micro['Semantic_Flaw'].value_counts().head(5).reset_index()
    semantic_micro_blindspots.columns = ['Semantic Flaw (Merged)', 'Times Missed by Humans but Caught by LLM']
    
    print("\n--- TOP 5 LỖI CHI TIẾT (Micro - Đã gộp ngữ nghĩa) bị bỏ sót nhiều nhất ---")
    print(semantic_micro_blindspots.to_string(index=False))

# ==========================================
# RQ6: DEAL-BREAKERS (NHỮNG LỖI TỬ HUYỆT) (DONE)
# ==========================================
print("\n" + "="*70)
print("RQ6 DRILL-DOWN: TOP LỖI CHI TIẾT (ĐÃ GỘP NGỮ NGHĨA) TRONG TỪNG NHÓM MACRO")
print("="*70)

# 1. Gom nhóm Micro theo Ngữ nghĩa
semantic_macro_to_micro = {}

for raw_flaw, weight in dealbreakers_micro.items():
    macro_topic = categorize_label(raw_flaw)
    if macro_topic == "Other / Uncategorized": continue
        
    semantic_micro = get_semantic_micro_label(raw_flaw, macro_topic)
    
    if macro_topic not in semantic_macro_to_micro:
        semantic_macro_to_micro[macro_topic] = {}
        
    semantic_macro_to_micro[macro_topic][semantic_micro] = semantic_macro_to_micro[macro_topic].get(semantic_micro, 0) + weight

# 2. Sắp xếp và in kết quả
sorted_macro_topics = sorted(dealbreakers_macro.items(), key=lambda x: x[1], reverse=True)

for macro, total_macro_weight in sorted_macro_topics:
    if macro not in semantic_macro_to_micro: continue
        
    print(f"\n📦 NHÓM LỚN: {macro.upper()} (Tổng trọng số: {total_macro_weight:.1f})")
    print("-" * 60)
    
    # Sort sub-topics by weight
    sorted_semantic_micro = sorted(semantic_macro_to_micro[macro].items(), key=lambda x: x[1], reverse=True)
    
    for i, (flaw_name, flaw_weight) in enumerate(sorted_semantic_micro[:4]): # Lấy Top 4
        percent_in_macro = (flaw_weight / total_macro_weight) * 100 if total_macro_weight > 0 else 0
        print(f"   {i+1}. {flaw_name}")
        print(f"      ↳ Trọng số: {flaw_weight:.1f} (Chiếm {percent_in_macro:.1f}% của nhóm)")