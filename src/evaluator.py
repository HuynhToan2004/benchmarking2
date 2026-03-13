

import json
from openai import OpenAI
from typing import Dict, List, Set, Tuple

class ReviewEvaluatorPipeline:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
        # Lưu trữ prompt (Nên load từ file text để code gọn gàng)
        self.prompt_step1 = """
        SYSTEM PROMPT (STEP 1):
You are an expert meta-reviewer for top-tier computer science conferences (e.g., ICLR, NeurIPS). Your task is to analyze raw review texts from multiple reviewers (both Human and AI) and consolidate their arguments into a structured list of unique "Micro-flaws".

For each unique weakness mentioned across the reviews, create a "Micro-flaw" object. 

CRITICAL RULES FOR GROUPING (STRICTLY ENFORCED):
You must avoid "Frankenstein" clustering (merging fundamentally different scientific issues just because they fall under the same broad Macro-Topic). Follow these rules:

1. CONCEPTUAL CONSISTENCY (Must Split): Arguments grouped into the same Micro-flaw MUST address the same fundamental conceptual, methodological, or experimental problem.
   - Example to SPLIT: If Reviewer A criticizes "scaling activation values to integers" and Reviewer B criticizes "using an LLM to validate the assessment", these are fundamentally different scientific issues. They MUST be separate Micro-flaws, even if both are "Experimental Design".
   - Example to SPLIT: A complaint about "limited dataset (MNIST)" and a complaint about "missing regularization details" are entirely different problems. DO NOT merge them.

2. ALLOWED AGGREGATION (Can Group): You MAY group arguments if they share the exact same nature or severity level, even if they point to different sections of the paper.
   - Example to GROUP: If Reviewer A says "broken references", Reviewer B says "Equation 7 dimensions don't match", and Reviewer C says "Section 3 is unclear", you CAN and SHOULD group them together under ONE Micro-flaw for "2. Clarity & Presentation - General writing & Clarity issues".

3. NO FORCED FIT: Do not force an argument into a mismatched Micro-flaw type just to group it. If an argument doesn't fit the existing types perfectly, use the "Other [Topic] Issues" category.
Categorize each Micro-flaw by selecting EXACTLY ONE Macro-topic and its corresponding Micro-flaw type from the hierarchical taxonomy below:

1. Novelty & Contribution
   - Limited Novelty
   - Incremental Contribution Only
   - Lack of Significance/Impact
   - Other Novelty Issues
2. Clarity & Presentation
   - General writing & Clarity issues
   - Unclear Math/ Notations
   - Poor Figures/Tables Quality
   - Grammar & Typos
   - Other Presentation Issues
3. Applicability, Scalability & Limitations
   - General Applicability Issues
   - Scalability & Complexity Concerns
   - Lack of Discussion on Limitations
   - Missing Broader Impact/Ethical Concerns
   - Other Limitation Issues
4. Experimental Design & Evaluation 
   - Missing/Weak Baselines
   - Insufficient Experimental Validation
   - Questionable Evaluation Metrics
   - Limited/Biased Datasets
   - Other Evaluation Issues
5. Related work & Citations
   - Missing Comparisons with Prior Work
   - Missing Relevant Citations
   - Missing Recent/Concurrent Works
   - Other Citation Issues
6. Methodology & Theoretical Soundness
   - Weak Theoretical Justification/Proofs
   - Methodological Flaws
   - Strong/Unrealistic Assumptions
   - Lack of Intuition/Justification
   - Other Methodology Issues
7. Reproducibility & Open Science
   - General Reproducibility Concerns
   - Insufficient Implementation Details
   - Missing Code/Data Repository
   - Other Reproducibility Issues
CRITICAL: DO NOT group arguments just because they belong to the same Macro-topic. Only group them if they point to the EXACT SAME specific error in the paper. If Human 1 talks about missing baseline A, and Human 2 talks about missing baseline B, they are TWO DIFFERENT Micro-flaws.
OUTPUT FORMAT:
You MUST output a valid JSON object strictly matching this schema:
{
  "micro_flaws": [
    {
      "flaw_id": "F01",
      "macro_topic": "<Macro-topic number and name> - <Micro-flaw type>",
      "core_summary": "<A concise 1-sentence summary of the weakness>",
      "raw_arguments": {
        "<EXACT_ID_FROM_INPUT_1>": "<Exact quote>",
        "<EXACT_ID_FROM_INPUT_2>": "<Exact quote>"
      }
    }
  ]
}
CRITICAL INSTRUCTION: The keys inside "raw_arguments" MUST exactly match the reviewer IDs provided in the input (e.g., "Human_1", "Human_2", "LLM_Reviewer"). DO NOT rename them to "Reviewer_ID_1" or anything else.
EXAMPLE OF MACRO_TOPIC FIELD:
"macro_topic": "4. Experimental Design & Evaluation - Missing/Weak Baselines"""
        self.prompt_step2 = """
SYSTEM PROMPT (STEP 2):
You are a strict and objective Meta-Reviewer in a top-tier Computer Science conference. You will be provided with the FULL TEXT of a submitted scientific paper and a JSON list of "Micro-flaws" raised by various reviewers.

Your task is to independently verify each Micro-flaw against the paper's text.
For EACH Micro-flaw, answer two questions based STRICTLY on the paper's content:
1. is_valid (True/False): Does this flaw actually exist in the paper? Is the reviewer's argument factually correct? (Return False if it's a hallucination, a misunderstanding, or an unreasonable request).
2. severity ("Critical" / "Minor"): If valid, you MUST assign a strict severity label based on the predefined ontology below.

ONTOLOGY FOR SEVERITY MAPPING:
Assign CRITICAL if the flaw falls into these categories:
- Methodology & Theoretical Soundness: Weak Theoretical Justification/Proofs, Methodological Flaws, Strong/Unrealistic Assumptions, Lack of Intuition/Justification.
- Experimental Design & Evaluation: Missing/Weak Baselines, Insufficient Experimental Validation, Questionable Evaluation Metrics, Limited/Biased Datasets.
- Novelty & Contribution: Limited Novelty, Incremental Contribution Only, Lack of Significance/Impact.
- Applicability & Reproducibility (Severe): General Applicability Issues, Scalability & Complexity Concerns, General Reproducibility Concerns.
- Related Work (Severe): Missing Empirical Comparisons with Prior Work.

Assign MINOR if the flaw falls into these categories:
- Clarity & Presentation: General writing & Clarity issues, Unclear Math/Notations (ambiguous symbols, not fundamentally wrong math), Poor Figures/Tables Quality, Grammar & Typos.
- Applicability & Limitations (Textual): Lack of Discussion on Limitations, Missing Broader Impact/Ethical Concerns.
- Related Work & Citations: Missing Relevant Citations, Missing Recent/Concurrent Works.
- Reproducibility & Open Science (Documentation): Insufficient Implementation Details (e.g., missing hyperparameters in the appendix), Missing Code/Data Repository.

CRUCIAL RULE: Do not guess. If fixing the flaw requires the authors to run new experiments or correct core mathematical equations, it is CRITICAL. If it only requires editing the text, formatting, or adding a reference, it is MINOR. If is_valid is False, set severity to "None".

OUTPUT FORMAT (Strict JSON):
{
  "evaluations": {
    "F01": {
      "is_valid": true,
      "severity": "Critical"
    },
    "F02": {
      "is_valid": true,
      "severity": "Minor"
    }
  }
}"""

    # def step1_atomize_and_group(self, human_reviews: Dict[str, str], llm_review: str) -> dict:
    #     """
    #     Input: 
    #         human_reviews: dict format {"Human_1": "text...", "Human_2": "text..."}
    #         llm_review: string chứa review của LLM
    #     """
    #     # Trộn input
    #     input_text = ""
    #     for reviewer_id, review_text in human_reviews.items():
    #         input_text += f"\n\n[REVIEWER: {reviewer_id}]\n{review_text}"
    #     input_text += f"\n\n[REVIEWER: LLM_Reviewer]\n{llm_review}"

    #     response = self.client.chat.completions.create(
    #         model="gpt-5-mini",
    #         response_format={"type": "json_object"}, # Ép trả về chuẩn JSON
    #         messages=[
    #             {"role": "system", "content": self.prompt_step1},
    #             {"role": "user", "content": f"Here are the reviews to analyze:\n{input_text}"}
    #         ],
    #         # temperature=0.1,
    #     )
    #     return json.loads(response.choices[0].message.content)

    # def step2_judge_flaws(self, paper_text: str, micro_flaws_json: dict) -> dict:
    #     """
    #     Input: Text của bài báo gốc và file JSON kết quả từ Step 1
    #     """
    #     flaws_str = json.dumps(micro_flaws_json, indent=2)
    #     input_text = f"[PAPER TEXT]\n{paper_text[:20000]}...\n\n[MICRO-FLAWS]\n{flaws_str}" # Cắt bớt nếu paper quá dài so với context window

    #     response = self.client.chat.completions.create(
    #         model="gpt-4o", # Nên dùng model mạnh hơn cho bước này
    #         response_format={"type": "json_object"},
    #         messages=[
    #             {"role": "system", "content": self.prompt_step2},
    #             {"role": "user", "content": input_text}
    #         ],
    #         temperature=0.0,
    #     )
    #     return json.loads(response.choices[0].message.content)
    def step1_atomize_and_group(self, human_reviews: Dict[str, str], llm_review: str) -> dict:
        """
        Gộp và phân tách các luận điểm đánh giá (Sử dụng gpt-5-mini)
        """
        input_text = ""
        for reviewer_id, review_text in human_reviews.items():
            input_text += f"\n\n[REVIEWER: {reviewer_id}]\n{review_text}"
        input_text += f"\n\n[REVIEWER: LLM_Reviewer]\n{llm_review}"

        print("  [INFO] Calling gpt-5-mini for Step 1 (Atomize & Group)...")
        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            response_format={"type": "json_object"}, 
            messages=[
                {"role": "system", "content": self.prompt_step1},
                {"role": "user", "content": f"Here are the reviews to analyze:\n{input_text}"}
            ],
            # temperature=0.1, # Giữ temperature thấp để JSON ổn định
        )
        return json.loads(response.choices[0].message.content)

    def step2_judge_flaws(self, paper_text: str, micro_flaws_json: dict) -> dict:
        """
        Thẩm định lỗi và Gán nhãn Severity (Sử dụng gpt-5-mini)
        """
        flaws_str = json.dumps(micro_flaws_json, indent=2)
        
        # CẮT NGẮN CONTEXT: Đảm bảo không vượt quá giới hạn token của model mini
        # Bạn có thể tăng con số 30000 lên nếu gpt-5-mini hỗ trợ context window lớn hơn
        safe_paper_text = paper_text[:30000] 
        
        input_text = f"[PAPER TEXT]\n{safe_paper_text}\n\n[MICRO-FLAWS]\n{flaws_str}"

        print("  [INFO] Calling gpt-5-mini for Step 2 (Judge & Label Severity)...")
        response = self.client.chat.completions.create(
            model="gpt-5-mini", 
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.prompt_step2},
                {"role": "user", "content": input_text}
            ],
            # temperature=0.0, # Đặt bằng 0.0 để model chấm điểm khách quan và nhất quán tuyệt đối
        )
        return json.loads(response.choices[0].message.content)

class MetricsCalculator:
    """
    Class này nhận kết quả từ Step 1 (Flaws Grouping) và Step 2 (Judgement) 
    để tính toán các chỉ số thống kê toán học.
    """
    def __init__(self, micro_flaws_json: dict, evaluations_json: dict):
        self.flaws = micro_flaws_json.get("micro_flaws", [])
        self.evals = evaluations_json.get("evaluations", {})
        
        # Khởi tạo các tập hợp (Sets) Ground Truth
        self.G_all = set()
        self.G_critical = set()
        self.G_minor = set()
        
        # Phân loại Ground Truth dựa trên kết quả của LLM Judge
        for flaw_id, result in self.evals.items():
            if result.get("is_valid") is True:
                self.G_all.add(flaw_id)
                if result.get("severity") == "Critical":
                    self.G_critical.add(flaw_id)
                elif result.get("severity") == "Minor":
                    self.G_minor.add(flaw_id)

    def get_reviewer_flaws(self, reviewer_id: str) -> set:
            """Lấy danh sách các flaw_id mà một reviewer cụ thể đã chỉ ra (Flexible Matching)"""
            detected_flaws = set()
            for flaw in self.flaws:
                raw_args = flaw.get("raw_arguments", {})
                
                for key in raw_args.keys():
                    # Xử lý cho LLM_Reviewer
                    if reviewer_id == "LLM_Reviewer" and ("llm" in key.lower() or "sea" in key.lower()):
                        detected_flaws.add(flaw["flaw_id"])
                        break
                    
                    # Xử lý cho Human_X (Ví dụ Input là "Human_1", nếu key là "Reviewer_ID_1" thì vẫn lấy)
                    elif reviewer_id.startswith("Human_"):
                        # Lấy con số định danh, ví dụ "1" từ "Human_1"
                        human_num = reviewer_id.split("_")[1] 
                        if human_num in key: 
                            detected_flaws.add(flaw["flaw_id"])
                            break
                            
            return detected_flaws

    def calculate_scores(self, reviewer_flaws: set) -> dict:
        """Tính toán Precision, Recall, F1 dựa trên toán học tập hợp"""
        # Precision
        true_positives = len(reviewer_flaws.intersection(self.G_all))
        precision = true_positives / len(reviewer_flaws) if len(reviewer_flaws) > 0 else 0.0
        
        # Recalls
        recall_critical = len(reviewer_flaws.intersection(self.G_critical)) / len(self.G_critical) if self.G_critical else 0.0
        recall_minor = len(reviewer_flaws.intersection(self.G_minor)) / len(self.G_minor) if self.G_minor else 0.0
        recall_overall = true_positives / len(self.G_all) if self.G_all else 0.0
        
        # F1-Score
        f1 = 2 * (precision * recall_overall) / (precision + recall_overall) if (precision + recall_overall) > 0 else 0.0
        
        return {
            "Precision": round(precision, 4),
            "Recall_Critical": round(recall_critical, 4),
            "Recall_Minor": round(recall_minor, 4),
            "Recall_Overall": round(recall_overall, 4),
            "F1_Score": round(f1, 4)
        }

    def generate_report(self, human_ids: List[str]) -> dict:
        """Tạo báo cáo so sánh xử lý Lỗ hổng 2 (Individual vs Collective)"""
        report = {}
        
        # 1. Điểm của LLM Reviewer
        llm_flaws = self.get_reviewer_flaws("LLM_Reviewer")
        report["LLM_Reviewer"] = self.calculate_scores(llm_flaws)
        
        # 2. Điểm của từng cá nhân Human và trung bình Human
        human_individual_scores = []
        human_collective_flaws = set()
        
        for h_id in human_ids:
            h_flaws = self.get_reviewer_flaws(h_id)
            report[h_id] = self.calculate_scores(h_flaws)
            human_individual_scores.append(report[h_id]["F1_Score"])
            
            # Gộp flaws cho Collective
            human_collective_flaws.update(h_flaws)
            
        # 3. Điểm của Hội đồng Human (Collective)
        report["Human_Collective"] = self.calculate_scores(human_collective_flaws)
        
        # 4. Tính Macro-Average F1 của Human để so sánh công bằng 1vs1
        report["Human_Average_F1"] = round(sum(human_individual_scores) / len(human_individual_scores) if human_individual_scores else 0, 4)
        
        return report

# --- CÁCH SỬ DỤNG TRONG PIPELINE CHÍNH ---
if __name__ == "__main__":
    # Khởi tạo
    pipeline = ReviewEvaluatorPipeline(api_key="YOUR_OPENAI_API_KEY")
    
    # Giả lập dữ liệu đã load từ hàm utils của bạn
    paper_content = "Toàn bộ text của bài báo..."
    human_reviews_dict = {
        "Human_1": "The baseline is missing...",
        "Human_2": "Equation 3 is wrong, and there are many typos."
    }
    llm_review_text = "It fails to compare with standard baselines. There are spelling mistakes."
    
    # Chạy Pipeline
    print("Running Step 1: Atomize & Group...")
    step1_output = pipeline.step1_atomize_and_group(human_reviews_dict, llm_review_text)
    
    print("Running Step 2: Evaluating Ground Truth...")
    step2_output = pipeline.step2_judge_flaws(paper_content, step1_output)
    
    print("Calculating Metrics...")
    calculator = MetricsCalculator(step1_output, step2_output)
    final_report = calculator.generate_report(human_ids=["Human_1", "Human_2"])
    
    print(json.dumps(final_report, indent=2))