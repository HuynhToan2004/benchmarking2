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

def load_paper_mmd(paper_id, mmd_folder):
    """Đọc file nội dung bài báo (.mmd) từ paper_nougat_mmd folder"""
    mmd_path = os.path.join(mmd_folder, f"{paper_id}.mmd")
    if os.path.exists(mmd_path):
        with open(mmd_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print(f"⚠️ Warning: Missing .mmd file for {paper_id}")
        return None

def get_paper_pairs(human_folder, sea_folder):
    """
    Tìm các cặp file (json, txt) trùng tên paper_id.
    """
    human_files = glob.glob(os.path.join(human_folder, "*.json"))
    pairs = []
    
    for h_path in human_files:
        basename = os.path.basename(h_path)
        paper_id = os.path.splitext(basename)[0]
        llm_path = os.path.join(sea_folder, f"{paper_id}.txt")
        
        if os.path.exists(llm_path):
            pairs.append((paper_id, h_path, llm_path))
        else:
            print(f"⚠️ Warning: Missing LLM review for {paper_id}")
            
    return pairs

def format_human_review_text(review_obj):
    """
    Chỉ trích xuất các phần liên quan đến PHẢN BIỆN.
    """
    text_parts = []
    if "Summary" in review_obj:
        text_parts.append(f"### Summary:\n{review_obj['Summary']}")
    if "Weaknesses" in review_obj:
        text_parts.append(f"### Weaknesses:\n{review_obj['Weaknesses']}")
    if "Questions" in review_obj:
        text_parts.append(f"### Questions:\n{review_obj['Questions']}")
    return "\n\n".join(text_parts)