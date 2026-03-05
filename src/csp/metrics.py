import math
from collections import defaultdict
from typing import List, Dict
from src.csp.config import SEVERITY_MAP

EPSILON = 1e-6

def calculate_nsr(arguments: List[Dict], paper_decision: str) -> float:
    sum_tokens_noise = 0
    sum_tokens_signal = 0

    for arg in arguments:
        severity = arg.get("severity")
        count = len(arg.get("content", "").strip().split())
        print(f"[Debug] Argument Severity: {severity}, Token Count: {count}")
        if severity == "Minor":
            sum_tokens_noise += count
        elif severity in ["Fatal", "Major"]:
            sum_tokens_signal += count
            
    # --- LOGIC XỬ LÝ CHIA CHO 0 THÔNG MINH ---
    if sum_tokens_signal == 0:
        # Nếu bài báo được ACCEPT, việc không tìm ra lỗi Major là ĐÚNG
        # -> NSR = 0 (Tuyệt vời, không phải là nhiễu, mà là sự im lặng đúng đắn)
        if "Accept" in paper_decision or "accept" in paper_decision.lower():
            return 0.0
            
        # Nếu bài báo bị REJECT, mà không tìm ra lỗi -> Reviewer KÉM
        # -> NSR = MAX (Phạt nặng)
        else:
            return 100.0 
        
    nsr = sum_tokens_noise / sum_tokens_signal
    return round(min(nsr, 100.0), 4)

def calculate_cps(arguments: List[Dict]) -> float:
    """
    Tính CPS với logic 'Local Rank' (Rank cục bộ theo từng phần).
    Rank sẽ được reset về 1 mỗi khi đổi sang Section khác.
    """
    cps = 0.0
    # Ví dụ: {'Summary': 0, 'Weaknesses': 0, 'Questions': 0}
    section_rank_counter = defaultdict(int)

    # Duyệt qua danh sách (Lưu ý: Danh sách này phải giữ đúng thứ tự xuất hiện từ trên xuống dưới)
    for arg in arguments:
        section = arg.get("section", "Unknown") # Mặc định là Unknown nếu lỗi
        severity_label = arg.get("severity", "None")
        weight = SEVERITY_MAP.get(severity_label, 0.0)

        if section in ["Paper Decision", "Decision", "Conclusion"]:
            continue
        # Bỏ qua None (Khen ngợi/Tóm tắt) -> Không tính vào rank
        # (Hoặc bạn có thể quyết định: Khen ngợi vẫn chiếm 1 vị trí rank? 
        #  Thường thì ta nên bỏ qua để công bằng cho các luận điểm phê bình).
        if weight == 0.0:
            continue

        # Tăng rank cho section hiện tại
        section_rank_counter[section] += 1
        current_rank = section_rank_counter[section]
        
        # Công thức CPS
        denominator = math.log2(current_rank + 1)
        cps += weight / denominator
        print(f"[Debug] Argument Section: {section}, Severity: {severity_label}, Weight: {weight}, Rank: {current_rank}, Partial CPS: {weight / denominator}")
    return round(cps, 4)