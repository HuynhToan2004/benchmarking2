# main_cfi.py
import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv

# Import các module (giữ nguyên import của bạn)
from src.gemini_client import GeminiClient
from src.utils import get_paper_pairs, load_human_meta_json, load_llm_txt
from src.cfi.extraction import extract_flaws
from src.cfi.metrics import calculate_flaw_weights, calculate_reviewer_performance

# --- CONFIG ---
HUMAN_DIR = r"data\Human_and_meta_reviews"
SEA_DIR = r"data\SEA_reviews"
OUTPUT_DIR = r"output_cfi"
MASTER_JSONL_FILE = os.path.join(OUTPUT_DIR, "cfi_results_detailed.jsonl")

def append_to_jsonl(data, filepath):
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def main():
    load_dotenv()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = GeminiClient(temperature=0.0)

    # Resume Logic
    processed_ids = set()
    if os.path.exists(MASTER_JSONL_FILE):
        with open(MASTER_JSONL_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try: processed_ids.add(json.loads(line)['paper_id'])
                except: pass

    pairs = get_paper_pairs(HUMAN_DIR, SEA_DIR)
    pairs_to_run = [p for p in pairs if p[0] not in processed_ids]
    print(f"Papers to process: {len(pairs_to_run)}")

    for paper_id, human_path, llm_path in tqdm(pairs_to_run):
        try:
            # A. Load
            full_json = load_human_meta_json(human_path)
            llm_text = load_llm_txt(llm_path)
            human_reviews = full_json.get("reviews", [])
            meta_review = full_json.get("Meta review", {})

            # B. Extract
            flaw_dict = extract_flaws(client, llm_text, human_reviews, meta_review)

            # C. Metrics
            flaw_weights = calculate_flaw_weights(flaw_dict, meta_weight_bonus=1.0)
            reviewer_scores = calculate_reviewer_performance(flaw_dict, flaw_weights)

            # --- DEBUG OUTPUT ---
            print(f"\n[Paper: {paper_id}]")
            print(f"   + Reviewers Found: {list(reviewer_scores.keys())}")
            print(f"   + Scores: {json.dumps(reviewer_scores, indent=2)}")
            # --------------------

            # D. Save
            res = {
                "paper_id": paper_id,
                "scores": reviewer_scores,
                "flaw_weights": flaw_weights,
                "details": flaw_dict
            }
            append_to_jsonl(res, MASTER_JSONL_FILE)
            time.sleep(1)

        except Exception as e:
            print(f"[Error] {paper_id}: {e}")

if __name__ == "__main__":
    main()

# import json

# input_path = r"D:\Code\Python\Research\LLMs_reviewer\Benchmarking\output_cfi\cfi_results_detailed.jsonl"
# output_path = r"D:\Code\Python\Research\LLMs_reviewer\Benchmarking\output_cfi\cfi_labels.txt"

# labels = set()

# with open(input_path, "r", encoding="utf-8") as f:
#     for line in f:
#         if not line.strip():
#             continue
#         data = json.loads(line)

#         # lấy key từ flaw_weights
#         if "flaw_weights" in data:
#             for k in data["flaw_weights"].keys():
#                 labels.add(k)

#         # lấy key từ details
#         if "details" in data:
#             for k in data["details"].keys():
#                 labels.add(k)

# # ghi ra file
# with open(output_path, "w", encoding="utf-8") as f:
#     for lb in sorted(labels):
#         f.write(lb + "\n")

# print(f"Extracted {len(labels)} unique labels -> {output_path}")