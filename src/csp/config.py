
SEVERITY_MAP = {
    "Fatal": 3.0,  
    "Major": 2.0,  
    "Minor": 1.0,  
    "None": 0.0    
}


TAXONOMY_DEFINITION = """
1. Fatal (Weight=3): Critical Flaws & Scientific Invalidity.
   * Definition: Fundamental errors that invalidate the paper's core claims. These make the results untrustworthy or the method mathematically/logically unsound.
   * STRICT CRITERIA: Must be supported by specific evidence (e.g., pointing out exactly which equation is wrong or how data leaked).
   * Examples:
     - "Equation 5 is mathematically incorrect because..."
     - "Data leakage: Test set labels used in training pre-processing."
     - "Strongly recommend rejection due to fatal flaw X."

2. Major (Weight=2): Significant Weaknesses (Actionable & Specific).
   * Definition: Serious issues that reduce quality but are fixable. Crucially, the critique must be **SPECIFIC**.
   * ANTI-BIAS RULE: If a critique is generic (e.g., "Novelty is limited", "Experiments are weak") WITHOUT citing specific missing baselines or prior work, DOWNGRADE it to 'Minor'.
   * Examples:
     - "Missing comparison with SOTA method [Author, Year]." (Specific -> Major)
     - "Ablation study is missing for component Z." (Specific -> Major)
     - "Generalization to dataset Y is unproven." (Specific -> Major)

3. Minor (Weight=1): Nitpicks, Clarifications & **Generic Critiques**.
   * Definition: Issues of presentation, formatting, simple questions, OR non-specific complaints that lack evidence.
   * Examples:
     - Presentation: Typos, grammar, blurry figures.
     - Clarification Questions: "What is the batch size?", "How did you choose hyperparam K?" (Unless it implies a fatal flaw).
     - **Generic/Vague Critiques**: "The contribution is marginal", "More experiments needed" (without saying which ones), "Lack of novelty" (without citations).

4. None (Weight=0): Non-Critique Content.
   * Definition: Summaries, Praise, Neutral statements, or procedural text.
"""

def get_analysis_prompt(review_text: str) -> str:
    return f"""
You are an expert Meta-Reviewer evaluating peer reviews. Your goal is to extract **High-Quality Critiques** and filter out noise.

Target: Extract critiques and classify them by **Section** and **Severity**.

Strict Taxonomy:
{TAXONOMY_DEFINITION}

Instructions:
1. **Context Scan:** Identify the section context (SUMMARY, WEAKNESSES, QUESTIONS).
2. **Extraction & Filtering:** Extract distinct arguments.
   - Ignore pure praise or summaries (Label as 'None').
   - If a 'Fatal' flaw appears in the Summary, capture it.
3. **Severity Assignment (ANTI-BIAS PROTOCOL):**
   - **The "Evidence" Rule:** Does the critique provide specific evidence (e.g., names of missing baselines, specific math errors)? 
     - YES -> Can be Fatal or Major.
     - NO (Vague/Generic) -> Must be **Minor**.
   - **The "Question" Rule:** Default all "Questions" to **Minor**. Only promote to Major/Fatal if the question exposes a logical fallacy or fundamental gap (e.g., "Why does the proof contradict Theorem 2?").
4. **Ordering:** Maintain the exact original order of appearance.

Output JSON:
{{
  "arguments": [
    {{
      "section": "Weaknesses", 
      "severity": "Major",
      "content": "Missing comparison with baseline X [Smith et al. 2023]."
    }},
    {{
      "section": "Weaknesses",
      "severity": "Minor",
      "content": "The paper lacks novelty (No specific prior work cited)."
    }}
  ]
}}

Review Text:
\"\"\"
{review_text}
\"\"\"
""".strip()