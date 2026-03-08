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
import re
def load_paper_mmd(paper_id, mmd_folder):
    """
    Đọc file nội dung bài báo (.mmd).
    Tối ưu: Cắt bỏ "## References" bằng string split siêu tốc, 
    nhưng vẫn quét và giữ lại Appendix ở phía sau.
    """
    mmd_path = os.path.join(mmd_folder, f"{paper_id}.mmd")
    if not os.path.exists(mmd_path):
        print(f"⚠️ Warning: Missing .mmd file for {paper_id}")
        return None
        
    with open(mmd_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Sử dụng hàm split cơ bản: Nhanh hơn và chính xác 100% theo format của bạn
    parts = content.split("## References")
    
    # Nếu không có chữ "## References" nào (mảng parts chỉ có 1 phần tử)
    if len(parts) == 1:
        return content
        
    # Nội dung chính là phần đầu tiên trước "## References"
    core_content = parts[0]
    
    # Nội dung đuôi là tất cả những gì nằm sau "## References"
    # Dùng join trong trường hợp hiếm hoi có nhiều chữ "## References" trong text
    tail_content = "## References".join(parts[1:])
    
    # Quét xem trong phần đuôi có Phụ lục không
    appx_pattern = r'\n#+\s*(Appendix|Appendices|Supplementary)\b'
    appx_match = re.search(appx_pattern, tail_content, re.IGNORECASE)
    
    if appx_match:
        # Nếu có Appendix, nối nó vào ngay sau phần core_content
        appendix_content = tail_content[appx_match.start():]
        core_content += "\n\n" + appendix_content
        print(f"  -> Đã tối ưu: Cắt bỏ References, giữ lại Appendix cho bài {paper_id}")
    else:
        print(f"  -> Đã tối ưu: Cắt bỏ hoàn toàn đuôi References cho bài {paper_id}")
        
    return core_content
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