# main.py
import os
import json
from src.utils import (
    load_human_meta_json, 
    load_llm_txt, 
    load_paper_mmd, 
    get_paper_pairs, 
    format_human_review_text
)
# Giả sử class ReviewEvaluatorPipeline và MetricsCalculator bạn đã lưu trong file evaluator.py
from src.evaluator import ReviewEvaluatorPipeline, MetricsCalculator
# Sửa lại dòng import này ở đầu file của bạn
from src.cfi.metrics import DecoupledMetricsCalculator
# --- CẤU HÌNH ĐƯỜNG DẪN TỪ HỆ THỐNG CỦA BẠN ---
HUMAN_FOLDER = r".\data\Human_and_meta_reviews" 
SEA_FOLDER = r".\data\SEA_reviews"
MMD_FOLDER = r".\data\paper_nougat_mmd"
OUTPUT_DIR = r".\output_cfi"
API_KEY = "dien api day"

def process_single_paper(paper_id, h_path, llm_path, pipeline):
    print(f"\n--- Đang xử lý Paper ID: {paper_id} ---")
    
    # 1. Load nội dung bài báo (.mmd)
    paper_content = load_paper_mmd(paper_id, MMD_FOLDER)
    if not paper_content:
        return None
        
    # 2. Load và Format Human Reviews
    human_data = load_human_meta_json(h_path)
    human_reviews_dict = {}
    
    # Giả định human_data là một List các dict chứa review của từng người.
    human_list = human_data.get("reviews", []) if isinstance(human_data, dict) else human_data
    
    for idx, review_obj in enumerate(human_list):
        formatted_text = format_human_review_text(review_obj)
        if formatted_text.strip():
            human_reviews_dict[f"Human_{idx+1}"] = formatted_text
            
    human_ids = list(human_reviews_dict.keys())
            
    # 3. Load LLM Review
    llm_review_text = load_llm_txt(llm_path)
    
    # 4. Chạy Pipeline Step 1 (Atomize & Group)
    print(">> Step 1: LLM Atomizing and Grouping arguments...")
    step1_flaws = pipeline.step1_atomize_and_group(human_reviews_dict, llm_review_text)
    
    # 5. Chạy Pipeline Step 2 (LLM Judge)
    print(">> Step 2: LLM Judging True/False & Critical/Minor...")
    step2_evals = pipeline.step2_judge_flaws(paper_content, step1_flaws)
    
    # 6. Tính toán Metrics bằng công thức Decoupled mới
    print(">> Step 3: Calculating Decoupled Metrics...")
    
    # Tính N (Tổng số reviewer) = Số lượng Human + 1 (LLM)
    total_reviewers = len(human_ids) + 1
    
    # Gọi class mới
    calculator = DecoupledMetricsCalculator(
        micro_flaws_json=step1_flaws, 
        evaluations_json=step2_evals,
        total_reviewers_count=total_reviewers
    )
    
    # Sử dụng hàm generate_final_report thay cho generate_report cũ
    report = calculator.generate_final_report(human_ids)
    
    # Trả về kết quả gộp để lưu lại
    return {
        "paper_id": paper_id,
        "micro_flaws": step1_flaws,
        "evaluations": step2_evals,
        "metrics_report": report
    }
def main():
    # Tạo thư mục output nếu chưa có
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Lấy danh sách các bài báo hợp lệ
    paper_pairs = get_paper_pairs(HUMAN_FOLDER, SEA_FOLDER)
    print(f"Tìm thấy {len(paper_pairs)} papers hợp lệ để xử lý.")
    
    # Khởi tạo Pipeline
    pipeline = ReviewEvaluatorPipeline(api_key=API_KEY)
    
    # Đường dẫn file output
    jsonl_output_path = os.path.join(OUTPUT_DIR, "all_papers_results.jsonl")
    
    # --- THÊM CƠ CHẾ RESUME ---
    processed_paper_ids = set()
    
    # Kiểm tra xem file đã tồn tại chưa để đọc lịch sử
    if os.path.exists(jsonl_output_path):
        with open(jsonl_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    # Đọc từng dòng JSON và lấy ID
                    data = json.loads(line)
                    if "paper_id" in data:
                        processed_paper_ids.add(data["paper_id"])
                except json.JSONDecodeError:
                    # Bỏ qua nếu có dòng nào bị lỗi format (ví dụ bị ngắt đột ngột lúc đang ghi)
                    continue
                    
    print(f"Đã tìm thấy {len(processed_paper_ids)} papers đã xử lý trong lịch sử. Sẽ bỏ qua các paper này.")
    
    # Vòng lặp chính
    for paper_id, h_path, llm_path in paper_pairs:
        # Bỏ qua nếu đã chạy rồi
        if paper_id in processed_paper_ids:
            print(f"⏩ Bỏ qua bài {paper_id} (Đã có kết quả)")
            continue
            
        try:
            # Gọi hàm xử lý từng bài (đã định nghĩa ở phần trước)
            result = process_single_paper(paper_id, h_path, llm_path, pipeline)
            
            if result:
                # Ghi nối tiếp (append) kết quả vào file JSONL ngay lập tức
                with open(jsonl_output_path, 'a', encoding='utf-8') as f:
                    # Chuyển dict thành chuỗi JSON trên 1 dòng duy nhất, không dùng indent
                    json_string = json.dumps(result, ensure_ascii=False)
                    f.write(json_string + "\n")
                
                # Thêm vào set để cập nhật trạng thái runtime
                processed_paper_ids.add(paper_id)
                
        except Exception as e:
            print(f"❌ Lỗi khi xử lý bài {paper_id}: {str(e)}")
            continue

    print(f"\n✅ Hoàn thành toàn bộ Pipeline! Kết quả được lưu tại: {jsonl_output_path}")

if __name__ == "__main__":
    main()




























































# # main_cfi.py
# import os
# import json
# import time
# from tqdm import tqdm
# from dotenv import load_dotenv

# # Import các module (giữ nguyên import của bạn)
# from src.gemini_client import GeminiClient
# from src.utils import get_paper_pairs, load_human_meta_json, load_llm_txt
# from src.cfi.extraction import extract_flaws
# from src.cfi.metrics import calculate_flaw_weights, calculate_reviewer_performance

# # --- CONFIG ---
# HUMAN_DIR = r"data\Human_and_meta_reviews"
# SEA_DIR = r"data\SEA_reviews"
# OUTPUT_DIR = r"output_cfi"
# MASTER_JSONL_FILE = os.path.join(OUTPUT_DIR, "cfi_results_detailed.jsonl")

# # def append_to_jsonl(data, filepath):
#     with open(filepath, 'a', encoding='utf-8') as f:
#         f.write(json.dumps(data, ensure_ascii=False) + '\n')

# def main():
#     load_dotenv()
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     client = GeminiClient(temperature=0.0)

#     # Resume Logic
#     processed_ids = set()
#     if os.path.exists(MASTER_JSONL_FILE):
#         with open(MASTER_JSONL_FILE, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try: processed_ids.add(json.loads(line)['paper_id'])
#                 except: pass

#     pairs = get_paper_pairs(HUMAN_DIR, SEA_DIR)
#     pairs_to_run = [p for p in pairs if p[0] not in processed_ids]
#     print(f"Papers to process: {len(pairs_to_run)}")

#     for paper_id, human_path, llm_path in tqdm(pairs_to_run):
#         try:
#             # A. Load
#             full_json = load_human_meta_json(human_path)
#             llm_text = load_llm_txt(llm_path)
#             human_reviews = full_json.get("reviews", [])
#             meta_review = full_json.get("Meta review", {})

#             # B. Extract
#             flaw_dict = extract_flaws(client, llm_text, human_reviews, meta_review)

#             # C. Metrics
#             flaw_weights = calculate_flaw_weights(flaw_dict, meta_weight_bonus=1.0)
#             reviewer_scores = calculate_reviewer_performance(flaw_dict, flaw_weights)

#             # --- DEBUG OUTPUT ---
#             print(f"\n[Paper: {paper_id}]")
#             print(f"   + Reviewers Found: {list(reviewer_scores.keys())}")
#             print(f"   + Scores: {json.dumps(reviewer_scores, indent=2)}")
#             # --------------------

#             # D. Save
#             res = {
#                 "paper_id": paper_id,
#                 "scores": reviewer_scores,
#                 "flaw_weights": flaw_weights,
#                 "details": flaw_dict
#             }
#             append_to_jsonl(res, MASTER_JSONL_FILE)
#             time.sleep(1)

#         except Exception as e:
#             print(f"[Error] {paper_id}: {e}")

# if __name__ == "__main__":
#     main()

# # import json

# # input_path = r".\output_cfi\cfi_results_detailed.jsonl"
# # output_path = r".\output_cfi\cfi_labels.txt"

# # labels = set()

# # with open(input_path, "r", encoding="utf-8") as f:
# #     for line in f:
# #         if not line.strip():
# #             continue
# #         data = json.loads(line)

# #         # lấy key từ flaw_weights
# #         if "flaw_weights" in data:
# #             for k in data["flaw_weights"].keys():
# #                 labels.add(k)

# #         # lấy key từ details
# #         if "details" in data:
# #             for k in data["details"].keys():
# #                 labels.add(k)

# # # ghi ra file
# # with open(output_path, "w", encoding="utf-8") as f:
# #     for lb in sorted(labels):
# #         f.write(lb + "\n")

# # print(f"Extracted {len(labels)} unique labels -> {output_path}")


