# source/utils.py
import json
import os
import glob

def load_human_meta_json(filepath):
    """Đọc file JSON Human & Meta review"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_llm_txt(filepath):
    """Đọc file TXT LLM review"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def get_paper_pairs(human_folder, sea_folder):
    """
    Tìm các cặp file (json, txt) trùng tên paper_id.
    Giả định: paper_id.json và paper_id.txt
    """
    human_files = glob.glob(os.path.join(human_folder, "*.json"))
    pairs = []
    
    for h_path in human_files:
        basename = os.path.basename(h_path)
        paper_id = os.path.splitext(basename)[0]
        
        # Tìm file txt tương ứng
        llm_path = os.path.join(sea_folder, f"{paper_id}.txt")
        
        if os.path.exists(llm_path):
            pairs.append((paper_id, h_path, llm_path))
        else:
            print(f"⚠️ Warning: Missing LLM review for {paper_id}")
            
    return pairs
def format_human_review_text(review_obj):
    """
    Chỉ trích xuất các phần liên quan đến PHẢN BIỆN để đánh giá Prioritization.
    Thứ tự ghép string rất quan trọng vì nó ảnh hưởng đến Rank trong CPS.
    """
    text_parts = []
    
    # 1. Summary: Nơi thể hiện ấn tượng đầu tiên. 
    # Nếu lỗi Fatal nằm ở đây -> Rank 1 (Điểm CPS cực cao).
    if "Summary" in review_obj:
        text_parts.append(f"### Summary:\n{review_obj['Summary']}")
        
    # 2. Weaknesses: Trọng tâm của bài review.
    if "Weaknesses" in review_obj:
        text_parts.append(f"### Weaknesses:\n{review_obj['Weaknesses']}")
        
    # 3. Questions: Phần bổ trợ, thường là Minor/Moderate.
    if "Questions" in review_obj:
        text_parts.append(f"### Questions:\n{review_obj['Questions']}")
    
    # Lưu ý: KHÔNG lấy Strengths, Rating, Confidence...
    
    return "\n\n".join(text_parts)

def format_meta_review_text(meta_obj):
    """
    Format cho Meta Review (Ground Truth)
    """
    text_parts = []
    if "Metareview" in meta_obj:
        text_parts.append(f"### Meta-Review Decision:\n{meta_obj['Metareview']}")
    if "Justification For Why Not Higher Score" in meta_obj:
        text_parts.append(f"### Key Rejection Reasons:\n{meta_obj['Justification For Why Not Higher Score']}")
        
    return "\n\n".join(text_parts)