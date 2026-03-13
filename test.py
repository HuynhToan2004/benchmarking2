import os
import json
from collections import Counter

def get_top_5_flaws_in_rejected_papers(jsonl_path, reviews_dir):
    flaw_counter = Counter()
    rejected_count = 0

    # Đọc file jsonl chứa dữ liệu lỗi
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            paper_id = data.get("paper_id")
            flaw_weights = data.get("flaw_weights", {})
            
            # Khớp với file quyết định trong folder Human_and_meta_reviews
            review_file = os.path.join(reviews_dir, f"{paper_id}.json")
            if os.path.exists(review_file):
                with open(review_file, 'r', encoding='utf-8') as rf:
                    review_data = json.loads(rf.read())
                    decision = review_data.get("Decision", "")
                    
                    # Lọc các bài có quyết định là Reject
                    if "Reject" in decision:
                        rejected_count += 1
                        # Đếm các lỗi có xuất hiện (weight > 0)
                        for flaw_name, weight in flaw_weights.items():
                            if weight > 0:
                                flaw_counter[flaw_name] += 1
                                
    print(f"Total number of Rejected papers analyzed: {rejected_count}")
    print("TOP 5 FLAWS MOST COMMON IN REJECTED PAPERS:")
    print("-" * 50)
    
    # In ra 5 lỗi phổ biến nhất
    top_5 = flaw_counter.most_common(5)
    for rank, (flaw, count) in enumerate(top_5, 1):
        print(f"{rank}. {flaw}: Appears {count} times  ({count/rejected_count*100:.1f}%)")

# Thay bằng đường dẫn thực tế trên máy bạn
jsonl_path = r"D:\Code\Python\Research\LLMs_reviewer\Benchmarking\output_cfi\cfi_results_detailed.jsonl"
reviews_dir = r"D:\Code\Python\Research\LLMs_reviewer\Benchmarking\data\Human_and_meta_reviews"

get_top_5_flaws_in_rejected_papers(jsonl_path, reviews_dir)