import json
from collections import defaultdict

# 1. Copy hàm categorize_label (Bộ quy tắc 7 Topic tôi vừa gửi ở câu trước) vào đây
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
            
    # Ưu tiên 1 -> 7
    if any(k in label for k in ["ambiguit", "caption", "clarif", "clarity", "confusing", "difficulty", "editorial", "english", "equation", "error", "explanation", "exposition", "figure", "fig ", "formatting", "grammar", "illustration", "inaccurate", "incorrect", "misleading", "inconsistent", "intuition", "notation", "organization", "overclaim", "overstate", "presentation", "readability", "structure", "table", "terminology", "typo", "visual", "wording", "writing"]) or ("detail" in label and "implementation" not in label): return "Clarity & Presentation"
    if any(k in label for k in ["anonymity", "availability", "code", "empty repository", "implementation", "open source", "release", "reproducibility"]): return "Reproducibility & Open Science"
    if any(k in label for k in ["citation", "concurrent work", "contextualization", "exclusion of", "existing", "literature", "omission of", "outdated", "prior work", "prior art", "reference", "related work"]): return "Related Work & Citations"
    if any(k in label for k in ["advantage", "benefit", "contribution", "incremental", "innovation", "marginal", "novelty", "obvious", "originality", "significance", "trivial", "uniqueness"]): return "Novelty & Contribution"
    if any(k in label for k in ["applicability", "broader impact", "complexity", "computational", "constraint", "cost", "discussion", "efficiency", "ethical", "extension", "flops", "future work", "generaliz", "gpu", "hardware", "hallucination", "handling", "latency", "limitation", "memory", "ood", "out-of-distribution", "overhead", "overfitting", "underfitting", "practical", "privacy", "real-world", "robustness", "runtime", "scalab", "scale", "scope", "societal", "speedup", "time"]): return "Applicability, Scalability & Limitations"
    if any(k in label for k in ["ablation", "accuracy", "analysis", "baseline", "benchmark", "comparison", "effectiveness", "empirical", "evaluation", "experiment", "fairness", "hyperparameter", "metric", "performance", "qualitative", "quantitative", "result", "sample", "validation", "zero-shot", "few-shot"]) or ("data" in label and "availability" not in label) or ("dataset" in label): return "Experimental Design & Evaluation"
    if any(k in label for k in ["algorithm", "architecture", "assumption", "backbone", "bound", "choice of", "convergence", "derivation", "equivalence", "expressiv", "formulation", "gap between", "guarantee", "heuristic", "hypothesis", "identifiability", "justification", "loss function", "method", "optimization", "parameter", "proof", "role of", "theoretical", "theory", "topology", "variance"]): return "Methodology & Theoretical Soundness"
    
    return "Other / Uncategorized"

# 2. Hai hàm tính điểm của bạn (giữ nguyên)
def calculate_flaw_weights(flaw_dict, meta_weight_bonus=1.0):
    flaw_weights = {}
    for flaw_name, reviewers_map in flaw_dict.items():
        score = 0.0
        for reviewer_id, quotes in reviewers_map.items():
            if quotes and len(quotes) > 0:
                score += 1.0
                if "Meta" in reviewer_id:
                    score += meta_weight_bonus
        flaw_weights[flaw_name] = score
    return flaw_weights

def calculate_reviewer_performance(flaw_dict, flaw_weights):
    reviewer_scores = defaultdict(float)
    for flaw_name, reviewers_map in flaw_dict.items():
        w = flaw_weights.get(flaw_name, 0.0)
        if w == 0: continue
        for reviewer_id, quotes in reviewers_map.items():
            if quotes and len(quotes) > 0:
                reviewer_scores[reviewer_id] += w
    return dict(reviewer_scores)

# 3. Kịch bản chạy chính
def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            paper_data = json.loads(line)
            
            old_details = paper_data.get("details", {})
            new_details = {}
            
            # --- MAP & MERGE ---
            for old_flaw, reviewers_dict in old_details.items():
                topic = categorize_label(old_flaw)
                
                if topic not in new_details:
                    new_details[topic] = {"LLM_Reviewer": [], "Meta_Reviewer": [], "Human_1": [], "Human_2": [], "Human_3": [], "Human_4": []}
                
                # Gộp quotes
                for reviewer, quotes in reviewers_dict.items():
                    if reviewer not in new_details[topic]:
                        new_details[topic][reviewer] = []
                    # Tránh thêm trùng lặp quote nếu có
                    for q in quotes:
                        if q not in new_details[topic][reviewer]:
                            new_details[topic][reviewer].append(q)
                            
            # --- RECALCULATE SCORES ---
            new_weights = calculate_flaw_weights(new_details)
            new_scores = calculate_reviewer_performance(new_details, new_weights)
            
            # --- CẬP NHẬT LẠI JSON ---
            paper_data["details"] = new_details
            paper_data["flaw_weights"] = new_weights
            paper_data["scores"] = new_scores
            
            # Ghi ra file mới
            f_out.write(json.dumps(paper_data, ensure_ascii=False) + "\n")

def get_paper_scores_flaw_weights(jsonl_file, output_file):
    paper_scores = {}
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            paper_id = data.get("paper_id", "unknown")
            scores = data.get("scores", {})
            flaw_weights = data.get("flaw_weights", {})
            paper_scores[paper_id] = {"scores": scores, "flaw_weights": flaw_weights}
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(paper_scores, ensure_ascii=False, indent=2) + "\n")

# Chạy thử
input_path = r"D:\Code\Python\Research\LLMs_reviewer\Benchmarking\output_cfi\cfi_results_detailed.jsonl"
output_path = r"D:\Code\Python\Research\LLMs_reviewer\Benchmarking\output_cfi\cfi_results_cleaned.jsonl"
# process_jsonl(input_path, output_path)
scores_output_path = r"D:\Code\Python\Research\LLMs_reviewer\Benchmarking\output_cfi\paper_scores_flaw_weights.json"
get_paper_scores_flaw_weights(output_path, scores_output_path)
print("Hoàn tất quy hoạch data!")