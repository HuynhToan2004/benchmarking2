# source/main.py
import os
import json
import time
from tqdm import tqdm

from gemini_client import GeminiClient, repair_json_with_model
from config import get_analysis_prompt
from metrics import calculate_nsr, calculate_cps
from utils import get_paper_pairs, load_human_meta_json, load_llm_txt, format_human_review_text

# path
HUMAN_DIR = r"data\Human_and_meta_reviews"
SEA_DIR = r"data\SEA_reviews"
OUTPUT_DIR = r"output"


MASTER_JSONL_FILE = os.path.join(OUTPUT_DIR, "all_results.jsonl")


def process_single_review(client, review_text, review_type, reviewer_id, paper_final_decision):
    """
    Hàm xử lý cho 1 bài review bất kỳ (Human/Meta/LLM)
    1. Gọi Gemini phân rã và đánh nhãn.
    2. Tính Metrics.
    """
    prompt = get_analysis_prompt(review_text)
    try:
        raw_response = client.generate_text(prompt)
        try:
            from gemini_client import extract_first_json_object
            json_str = extract_first_json_object(raw_response)
            parsed_data = json.loads(json_str)
        except:
            print(f"  Refining JSON for {review_type}...")
            parsed_data = repair_json_with_model(client, raw_response, "{'arguments': [...]}")
            
        arguments = parsed_data.get("arguments", [])
        nsr = calculate_nsr(arguments, paper_final_decision)
        cps = calculate_cps(arguments)
        
        return {
            "reviewer_id": reviewer_id,
            "type": review_type,
            "metrics": {"NSR": nsr, "CPS": cps},
            "analysis_details": arguments
        }
    except Exception as e:
        print(f"Error processing {reviewer_id}: {e}")
        return None
    

def append_to_jsonl(data, filepath):
    with open(filepath, 'a', encoding='utf-8') as f:
        json_line = json.dumps(data, ensure_ascii=False)
        f.write(json_line + '\n')


def main():
    # 1. Khởi tạo
    from dotenv import load_dotenv
    load_dotenv()
    client = GeminiClient(temperature=0.2)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # [TÙY CHỌN] Xóa file cũ nếu muốn chạy lại từ đầu. 
    # Nếu muốn chạy tiếp (resume) thì comment dòng này lại.
    if os.path.exists(MASTER_JSONL_FILE):
        print(f"⚠️ Found existing {MASTER_JSONL_FILE}. Appending to it...")
        # os.remove(MASTER_JSONL_FILE) # Bỏ comment nếu muốn xóa làm lại từ đầu

    # 2. Lấy danh sách bài báo
    paper_pairs = get_paper_pairs(HUMAN_DIR, SEA_DIR)
    print(f"Found {len(paper_pairs)} papers to process.")
    
    # 3. Vòng lặp chính qua từng Paper
    for paper_id, human_path, llm_path in tqdm(paper_pairs, desc="Processing Papers"):
        
        # --- Logic Resume (Tùy chọn) ---
        # Kiểm tra xem paper_id này đã có trong file jsonl chưa để bỏ qua
        with open(MASTER_JSONL_FILE, 'r', encoding='utf-8') as f_check:
            if f'"{paper_id}"' in f_check.read():
                continue 

       
        
        # --- A. Xử lý Input ---
        human_data = load_human_meta_json(human_path)
        llm_text = load_llm_txt(llm_path)
        # --- Get decision: reject, accept from paper meta review ---
        decision = human_data.get("Decision", "Unknown")
        paper_result = {
            "paper_id": paper_id,
            "Decision": decision,
            "reviews_evaluation": []
        }

        # --- B. Xử lý Meta Review ---
        if "Meta review" in human_data:
            meta_text = human_data["Meta review"].get("Metareview", "")
            print("[Debug] Meta Review Text:", meta_text)
            if meta_text:
                res = process_single_review(client, meta_text, "Meta", "Meta_Reviewer", decision)
                if res: paper_result["reviews_evaluation"].append(res)
                
        # --- C. Xử lý Human Reviews ---
        reviews_list = human_data.get("reviews", [])
        print(f"[Debug] Found {len(reviews_list)} Human Reviews.")
        for idx, review_obj in enumerate(reviews_list):
            h_text = format_human_review_text(review_obj)
            res = process_single_review(client, h_text, "Human", f"Human_{idx+1}",decision)
            if res: paper_result["reviews_evaluation"].append(res)
            
        # --- D. Xử lý LLM Review ---
        res_llm = process_single_review(client, llm_text, "LLM", "SEA_LLM", decision)
        if res_llm: paper_result["reviews_evaluation"].append(res_llm)
        
        # --- E. LƯU JSONL NGAY LẬP TỨC (QUAN TRỌNG) ---
        append_to_jsonl(paper_result, MASTER_JSONL_FILE)
        
        # (Tùy chọn) Vẫn lưu file lẻ json để debug nếu cần, không thì bỏ đi cũng được
        # output_file_single = os.path.join(OUTPUT_DIR, f"{paper_id}_eval.json")
        # with open(output_file_single, 'w', encoding='utf-8') as f:
        #    json.dump(paper_result, f, indent=4, ensure_ascii=False)
        
        # Sleep nhẹ
        time.sleep(2)

    print(f"✅ Completed! All results appended to {MASTER_JSONL_FILE}")
if __name__ == "__main__":
    from dotenv import load_dotenv
    main()
