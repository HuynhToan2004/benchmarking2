import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load Data
data = []
# Assuming 'data_str' contains the raw text you pasted. 
# In practice, you would load this from your .jsonl file.
raw_data = [
    # ... Paste your JSON objects here inside the list ...
    # For this example, I will assume 'json_objects' is a list of dicts loaded from your input
]

# (Placeholder for loading the data you provided in the prompt)
# For the user: simple copy your JSON lines into a file named 'cfi_results.jsonl'
try:
    with open(r'D:\Code\Python\Research\LLMs_reviewer\Benchmarking\output_cfi\cfi_results_detailed.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
except FileNotFoundError:
    print("Please save your data to 'cfi_results.jsonl' to run this script.")
    data = [] # Empty fallback

# 2. Process Data for Plotting
plot_rows = []

for entry in data:
    paper_id = entry['paper_id']
    scores = entry['scores']
    
    # LLM Score
    if 'LLM_Reviewer' in scores:
        plot_rows.append({
            'Paper ID': paper_id, 
            'Reviewer Type': 'LLM', 
            'Score': scores['LLM_Reviewer']
        })
    
    # Meta Score
    if 'Meta_Reviewer' in scores:
        plot_rows.append({
            'Paper ID': paper_id, 
            'Reviewer Type': 'Meta', 
            'Score': scores['Meta_Reviewer']
        })
        
    # Human Scores (Average)
    human_scores = [v for k, v in scores.items() if k.startswith('Human')]
    if human_scores:
        avg_human = np.mean(human_scores)
        plot_rows.append({
            'Paper ID': paper_id, 
            'Reviewer Type': 'Human (Avg)', 
            'Score': avg_human
        })

df = pd.DataFrame(plot_rows)

# 3. Create Visualization
if not df.empty:
    plt.figure(figsize=(12, 6))
    
    # Grouped Bar Chart
    sns.barplot(data=df, x='Paper ID', y='Score', hue='Reviewer Type', palette='viridis')
    
    plt.title('Flaw Severity Scores: LLM vs. Human vs. Meta', fontsize=14)
    plt.ylabel('Total Flaw Score (Weighted)', fontsize=12)
    plt.xlabel('Paper ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Reviewer')
    plt.tight_layout()
    plt.show()
    
    # 4. Correlation Analysis
    # Pivot table to get columns: LLM, Meta, Human (Avg)
    pivot_df = df.pivot(index='Paper ID', columns='Reviewer Type', values='Score')
    print("\n--- Correlation Matrix (Do LLM and Humans agree on which papers are bad?) ---")
    print(pivot_df.corr())
else:
    print("No data loaded. Check jsonl file.")