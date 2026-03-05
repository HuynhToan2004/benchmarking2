# src/cfi/metrics.py
from collections import defaultdict

def calculate_flaw_weights(flaw_dict, meta_weight_bonus=1.0):
    """
    Tính trọng số cho từng lỗi.
    Logic Consensus: Càng nhiều người (LLM, Human_1, Human_2...) nhắc tới thì điểm càng cao.
    """
    flaw_weights = {}
    
    for flaw_name, reviewers_map in flaw_dict.items():
        score = 0.0
        
        # Duyệt qua tất cả các reviewer trong map (Human_1, Human_2, LLM...)
        for reviewer_id, quotes in reviewers_map.items():
            # Nếu có trích dẫn -> Reviewer này đã phát hiện lỗi
            if quotes and len(quotes) > 0:
                # Cộng 1 điểm cơ bản
                score += 1.0
                
                # Nếu là Meta -> Cộng thêm bonus
                if "Meta" in reviewer_id:
                    score += meta_weight_bonus
                    
        flaw_weights[flaw_name] = score
        
    return flaw_weights

def calculate_reviewer_performance(flaw_dict, flaw_weights):
    """
    Tính điểm cho TỪNG Reviewer riêng biệt.
    Output: { "LLM_Reviewer": 5.0, "Human_1": 2.0, "Human_2": 4.0, ... }
    """
    # Dùng defaultdict để tự động tạo key mới nếu thấy Human mới
    reviewer_scores = defaultdict(float)
    
    for flaw_name, reviewers_map in flaw_dict.items():
        # Lấy trọng số của lỗi này (đã tính ở hàm trên)
        w = flaw_weights.get(flaw_name, 0.0)
        
        if w == 0: continue
        
        # Cộng điểm cho từng reviewer phát hiện ra lỗi này
        for reviewer_id, quotes in reviewers_map.items():
            if quotes and len(quotes) > 0:
                reviewer_scores[reviewer_id] += w
                
    return dict(reviewer_scores) # Convert về dict thường