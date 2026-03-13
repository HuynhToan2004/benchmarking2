# src/cfi/metrics.py
import json
from typing import List, Dict, Set

class DecoupledMetricsCalculator:
    """
    Class này tính toán hệ điểm Decoupled (Critical Score & Minor Score)
    dựa trên Consensus (C_i / N) và Severity do LLM Judge phân loại.
    """
    def __init__(self, micro_flaws_json: dict, evaluations_json: dict, total_reviewers_count: int):
        self.flaws = micro_flaws_json.get("micro_flaws", [])
        self.evals = evaluations_json.get("evaluations", {})
        self.N = total_reviewers_count
        
        # --- PART 1: TÍNH FLAW WEIGHT (W_i) ---
        # self.flaw_weights format: {"F01": {"weight": 0.8, "severity": "Critical"}, ...}
        self.flaw_weights = {}
        
        for flaw in self.flaws:
            flaw_id = flaw["flaw_id"]
            eval_data = self.evals.get(flaw_id, {})
            
            # Bỏ qua ngay lập tức nếu LLM Judge đánh giá lỗi này là Hallucination (False)
            if not eval_data.get("is_valid"):
                continue
                
            severity = eval_data.get("severity") # "Critical" hoặc "Minor"
            
            # Tính Consensus Count (C_i): Đếm số lượng reviewer có trích dẫn hợp lệ cho lỗi này
            raw_args = flaw.get("raw_arguments", {})
            C_i = sum(1 for quote in raw_args.values() if quote and len(quote.strip()) > 0)
            
            # Tính W_i = C_i / N
            W_i = C_i / self.N if self.N > 0 else 0.0
            
            self.flaw_weights[flaw_id] = {
                "weight": W_i,
                "severity": severity
            }

    def get_reviewer_flaws(self, reviewer_id: str) -> set:
        """Lấy danh sách các flaw_id mà một reviewer cụ thể đã chỉ ra."""
        detected_flaws = set()
        for flaw in self.flaws:
            raw_args = flaw.get("raw_arguments", {})
            for key in raw_args.keys():
                # Xử lý linh hoạt tên key (ví dụ: "LLM_Reviewer", "Human_1")
                if reviewer_id == "LLM_Reviewer" and ("llm" in key.lower() or "sea" in key.lower()):
                    detected_flaws.add(flaw["flaw_id"])
                    break
                elif reviewer_id.startswith("Human_"):
                    human_num = reviewer_id.split("_")[1] 
                    if human_num in key: 
                        detected_flaws.add(flaw["flaw_id"])
                        break
        return detected_flaws

    def calculate_reviewer_scores(self, reviewer_id: str) -> dict:
        """
        --- PART 2: TÍNH REVIEWER CAPABILITY SCORES (S_r) ---
        Cộng dồn W_i vào 2 rổ tách biệt: S_critical và S_minor.
        """
        s_critical = 0.0
        s_minor = 0.0
        
        detected_flaws = self.get_reviewer_flaws(reviewer_id)
        
        for flaw_id in detected_flaws:
            if flaw_id in self.flaw_weights:
                fw = self.flaw_weights[flaw_id]
                
                # Bỏ vào rổ (bucket) tương ứng
                if fw["severity"] == "Critical":
                    s_critical += fw["weight"]
                elif fw["severity"] == "Minor":
                    s_minor += fw["weight"]
                    
        return {
            "Reviewer_ID": reviewer_id,
            "Critical_Score": round(s_critical, 4),
            "Minor_Score": round(s_minor, 4),
            "Total_Valid_Flaws_Found": len(detected_flaws.intersection(self.flaw_weights.keys()))
        }

    def generate_final_report(self, human_ids: List[str]) -> dict:
        """Xuất báo cáo tổng kết so sánh LLM và Human"""
        report = {
            "Flaw_Weights_Summary": self.flaw_weights,
            "Reviewer_Rankings": []
        }
        
        # 1. Điểm của LLM
        llm_scores = self.calculate_reviewer_scores("LLM_Reviewer")
        report["Reviewer_Rankings"].append(llm_scores)
        
        # 2. Điểm của từng Human
        for h_id in human_ids:
            h_scores = self.calculate_reviewer_scores(h_id)
            report["Reviewer_Rankings"].append(h_scores)
            
        # Tự động Sort danh sách theo đúng Luật Lexicographical: 
        # Ưu tiên 1: Critical_Score (Giảm dần). Ưu tiên 2: Minor_Score (Giảm dần)
        report["Reviewer_Rankings"].sort(
            key=lambda x: (x["Critical_Score"], x["Minor_Score"]), 
            reverse=True
        )
        
        return report

# --- CÁCH SỬ DỤNG (Main Test) ---
if __name__ == "__main__":
    # Giả lập data output từ Step 1 và Step 2
    mock_step1 = {
        "micro_flaws": [
            {
                "flaw_id": "F01", 
                "raw_arguments": {"LLM_Reviewer": "quote", "Human_1": "quote"}
            },
            {
                "flaw_id": "F02", 
                "raw_arguments": {"Human_1": "quote", "Human_2": "quote", "Human_3": "quote", "LLM_Reviewer": "quote"}
            }
        ]
    }
    
    mock_step2 = {
        "evaluations": {
            "F01": {"is_valid": True, "severity": "Critical"},
            "F02": {"is_valid": True, "severity": "Minor"}
        }
    }
    
    # Giả sử hội đồng có 4 người: LLM_Reviewer, Human_1, Human_2, Human_3 -> N = 4
    N = 4 
    
    calculator = DecoupledMetricsCalculator(
        micro_flaws_json=mock_step1, 
        evaluations_json=mock_step2, 
        total_reviewers_count=N
    )
    
    final_report = calculator.generate_final_report(human_ids=["Human_1", "Human_2", "Human_3"])
    print(json.dumps(final_report, indent=2))