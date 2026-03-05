# src/cfi/extraction.py
import json
from src.gemini_client import GeminiClient, repair_json_with_model

# --- PROMPT NÂNG CẤP ---
# Thêm hướng dẫn cụ thể: Nếu không tìm thấy lỗi, hãy tìm bất kỳ bình luận tiêu cực nào.
EXTRACTION_PROMPT = """
You are an expert Area Chair. Your task is to extract "Weaknesses" or "Critiques" from the provided reviews.

### INPUT DATA:
{reviews_text}

### INSTRUCTIONS:
1. **Analyze Each Reviewer:** Look at LLM_Reviewer, Meta_Reviewer, and every Human_x.
2. **Identify Flaws:** Extract specific technical weaknesses, concerns, or questions raised by the reviewers.
3. **Consolidate:** Group similar points into a single "Flaw Name" (e.g., "Missing Baselines").
4. **Map Evidence (CRITICAL):** - You MUST attribute quotes to the correct reviewer ID.
   - If a reviewer mentions a flaw, you MUST include a short quote as evidence.
   - Do NOT halluncinate quotes. If a reviewer didn't say it, leave their list empty [].

### OUTPUT FORMAT (Strict JSON):
{{
    "Short_Flaw_Name_1": {{
        "LLM_Reviewer": ["quote..."],
        "Meta_Reviewer": ["quote..."],
        "Human_1": ["quote..."],
        "Human_2": []
    }},
    "Short_Flaw_Name_2": {{
        "LLM_Reviewer": [],
        "Meta_Reviewer": [],
        "Human_1": ["quote..."],
        "Human_2": ["quote..."]
    }}
}}
"""

def smart_get_text(obj, keys_to_try):
    """
    Hàm đệ quy tìm text cực mạnh, chấp nhận mọi cấu trúc OpenReview.
    """
    if not isinstance(obj, dict): return ""
    
    # 1. Ưu tiên tìm trong 'content' (Cấu trúc chuẩn OpenReview)
    if 'content' in obj and isinstance(obj['content'], dict):
        # Đệ quy vào trong content
        return smart_get_text(obj['content'], keys_to_try)
        
    # 2. Tìm các key mục tiêu ở level hiện tại
    candidates = []
    for k in keys_to_try: # ['review', 'weaknesses', 'summary', ...]
        val = obj.get(k)
        if val and isinstance(val, str) and len(val.strip()) > 10:
            # Nếu tìm thấy key xịn (vd: 'weaknesses'), trả về ngay
            if k in ['review', 'weaknesses', 'metareview', 'body']:
                return val
            candidates.append(val)
            
    # 3. Nếu không tìm thấy key xịn, ghép tất cả text tìm được
    if candidates:
        return "\n".join(candidates)

    # 4. Fallback: Tìm trong tất cả các value là string
    # (Dùng cho các cấu trúc lạ)
    fallback_text = []
    for v in obj.values():
        if isinstance(v, str) and len(v) > 50:
            fallback_text.append(v)
        elif isinstance(v, dict):
            child_text = smart_get_text(v, keys_to_try)
            if child_text: fallback_text.append(child_text)
            
    return "\n\n".join(fallback_text)

def extract_flaws(client: GeminiClient, llm_text: str, human_reviews: list, meta_review: dict) -> dict:
    # 1. Xây dựng Context
    combined_text = ""
    
    # --- LLM ---
    combined_text += f"=== ID: LLM_Reviewer ===\n{llm_text}\n\n"
    
    # --- Meta Review ---
    # Thêm nhiều key để bắt dính Meta review
    meta_keys = ['metareview', 'meta_review', 'recommendation', 'comment', 'summary']
    meta_content = smart_get_text(meta_review, meta_keys)
    
    if meta_content:
        combined_text += f"=== ID: Meta_Reviewer ===\n{meta_content}\n\n"
    else:
        # Debug để biết tại sao mất Meta
        print("   [DEBUG] Meta Review content is empty/not found.")

    # --- Human Reviews ---
    human_keys = ['review', 'main_review', 'body', 'content', 'weaknesses', 'summary', 'comments']
    
    if isinstance(human_reviews, list):
        for i, rev in enumerate(human_reviews):
            reviewer_id = f"Human_{i+1}"
            content = smart_get_text(rev, human_keys)
            
            if content:
                combined_text += f"=== ID: {reviewer_id} ===\n{content}\n\n"
            else:
                print(f"   [DEBUG] {reviewer_id} content is empty.")

    # 2. Call Gemini
    # print(f"   [DEBUG] Input Text Length: {len(combined_text)}") # Bỏ comment để debug
    
    prompt = EXTRACTION_PROMPT.format(reviews_text=combined_text)
    raw_response = client.generate_text(prompt)

    # 3. Parse Output
    try:
        from src.gemini_client import extract_first_json_object, json_loads_lenient
        json_str = extract_first_json_object(raw_response)
        return json_loads_lenient(json_str)
    except Exception as e:
        print(f"   [WARN] JSON Parse failed: {e}. Attempting repair...")
        # Schema hint để sửa lỗi
        hint = json.dumps({
            "Example_Flaw": {
                "LLM_Reviewer": [], "Meta_Reviewer": [], "Human_1": []
            }
        })
        return repair_json_with_model(client, raw_response, hint)