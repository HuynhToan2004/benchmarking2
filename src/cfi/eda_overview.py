import json
import pandas as pd

def load_cfi_data(jsonl_path):
    """
    Đọc file JSONL và chuyển đổi thành 3 DataFrames:
    1. papers_df: Thông tin cấp bài báo (scores).
    2. flaws_df: Thông tin cấp lỗi (flaw weights).
    3. mentions_df: Thông tin cấp trích dẫn (chi tiết ai nói gì).
    """
    papers_data = []
    flaws_data = []
    mentions_data = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            p_id = entry['paper_id']
            
            # 1. Papers Data
            # Flatten scores dictionary into columns
            paper_row = {'paper_id': p_id}
            paper_row.update(entry.get('scores', {}))
            papers_data.append(paper_row)

            # 2. Flaws & Mentions Data
            details = entry.get('details', {})
            weights = entry.get('flaw_weights', {})

            for flaw_name, reviewer_map in details.items():
                # Lấy weight (nếu không có thì default 0)
                w = weights.get(flaw_name, 0.0)
                
                # Thêm vào flaws_data
                flaws_data.append({
                    'paper_id': p_id,
                    'flaw_name': flaw_name,
                    'weight': w
                })

                # Thêm vào mentions_data
                for reviewer_id, quotes in reviewer_map.items():
                    if quotes and len(quotes) > 0: # Chỉ lấy nếu có quote
                        for quote in quotes:
                            mentions_data.append({
                                'paper_id': p_id,
                                'flaw_name': flaw_name,
                                'reviewer_id': reviewer_id,
                                'quote': quote
                            })

    # Convert to DataFrames
    papers_df = pd.DataFrame(papers_data)
    flaws_df = pd.DataFrame(flaws_data)
    mentions_df = pd.DataFrame(mentions_data)

    return papers_df, flaws_df, mentions_df


import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- CONFIG ---
INPUT_FILE = r"D:\Code\Python\Research\LLMs_reviewer\Benchmarking\output_cfi\cfi_results_detailed.jsonl" # Đường dẫn file JSONL của bạn
OUTPUT_DIR = r"D:\Code\Python\Research\LLMs_reviewer\Benchmarking\output_cfi\analysis_results"

def setup_style():
    """Thiết lập style cho biểu đồ đẹp hơn"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10

def plot_histogram(data, x_col, title, xlabel, filename, color='skyblue'):
    """Hàm vẽ histogram chung"""
    plt.figure()
    sns.histplot(data=data, x=x_col, kde=True, color=color, bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency (Papers)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")

def plot_top_horizontal(series, title, xlabel, filename, color='salmon'):
    """Hàm vẽ biểu đồ thanh ngang cho Top K"""
    plt.figure(figsize=(12, 8))
    sns.barplot(x=series.values, y=series.index, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")

def analyze_meta_coverage(papers_df, mentions_df):
    """Tính tỷ lệ paper có Meta Reviewer phát hiện lỗi"""
    total_papers = papers_df['paper_id'].nunique()
    
    # Lọc các mention từ Meta_Reviewer
    # Lưu ý: check cả 'Meta_Reviewer' và 'Meta_Review' đề phòng biến thể
    meta_mentions = mentions_df[
        mentions_df['reviewer_id'].str.contains("Meta", case=False, na=False)
    ]
    
    papers_with_meta_flaws = meta_mentions['paper_id'].nunique()
    ratio = (papers_with_meta_flaws / total_papers) * 100 if total_papers > 0 else 0
    
    return total_papers, papers_with_meta_flaws, ratio

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File input không tồn tại tại {INPUT_FILE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_style()

    print("--- Loading Data ---")
    papers_df, flaws_df, mentions_df = load_cfi_data(INPUT_FILE)
    print(f"Loaded: {len(papers_df)} papers, {len(flaws_df)} flaw instances, {len(mentions_df)} mentions.")

    # =================================================================
    # 1. Histogram: Số lượng lỗi trên mỗi bài báo (n_flaws per paper)
    # =================================================================
    # Group flaws_df theo paper_id và đếm
    flaws_count = flaws_df.groupby('paper_id').size().reset_index(name='n_flaws')
    # Merge lại với papers_df để đảm bảo những bài 0 lỗi cũng được tính (nếu có)
    papers_stats = papers_df[['paper_id']].merge(flaws_count, on='paper_id', how='left').fillna(0)
    
    plot_histogram(
        papers_stats, 
        'n_flaws', 
        'Distribution of Number of Flaws per Paper', 
        'Number of Flaws Identified',
        'hist_n_flaws.png',
        color='teal'
    )

    # =================================================================
    # 2. Histogram: Tổng trọng số lỗi trên mỗi bài báo (total_weight per paper)
    # =================================================================
    # Group flaws_df theo paper_id và tính tổng weight
    weight_sum = flaws_df.groupby('paper_id')['weight'].sum().reset_index(name='total_weight')
    papers_stats = papers_stats.merge(weight_sum, on='paper_id', how='left').fillna(0)

    plot_histogram(
        papers_stats, 
        'total_weight', 
        'Distribution of Total Flaw Weight per Paper', 
        'Total Flaw Weight (Severity)',
        'hist_total_weight.png',
        color='cornflowerblue'
    )

    # =================================================================
    # 3. Top 20 flaws xuất hiện nhiều nhất (Frequency by Paper Count)
    # =================================================================
    # Đếm số lần mỗi flaw_name xuất hiện
    top_freq_flaws = flaws_df['flaw_name'].value_counts().head(20)
    
    plot_top_horizontal(
        top_freq_flaws,
        'Top 20 Most Frequent Critical Flaws',
        'Number of Papers Containing Flaw',
        'top20_freq_flaws.png',
        color='indianred'
    )

    # =================================================================
    # 4. Top 20 flaws có trọng số TRUNG BÌNH cao nhất (Highest Avg Severity)
    # =================================================================
    # Chỉ tính những lỗi xuất hiện ít nhất N lần để tránh nhiễu (ví dụ xuất hiện 1 lần nhưng điểm cao)
    min_occurrence = 2
    flaw_stats = flaws_df.groupby('flaw_name')['weight'].agg(['mean', 'count'])
    flaw_stats_filtered = flaw_stats[flaw_stats['count'] >= min_occurrence]
    
    top_avg_weight_flaws = flaw_stats_filtered['mean'].sort_values(ascending=False).head(20)

    plot_top_horizontal(
        top_avg_weight_flaws,
        f'Top 20 Flaws with Highest Avg Weight (Min {min_occurrence} occurrences)',
        'Average Weight Score',
        'top20_avg_weight_flaws.png',
        color='orange'
    )

    # =================================================================
    # 5. Tỉ lệ paper có Meta Reviewer mention ít nhất 1 flaw
    # =================================================================
    total, meta_count, ratio = analyze_meta_coverage(papers_df, mentions_df)
    
    print("\n" + "="*40)
    print("STATISTICAL SUMMARY")
    print("="*40)
    print(f"Total Papers Processed: {total}")
    print(f"Avg Flaws per Paper:    {papers_stats['n_flaws'].mean():.2f}")
    print(f"Avg Weight per Paper:   {papers_stats['total_weight'].mean():.2f}")
    print("-" * 40)
    print(f"Papers with Meta-Reviewer Flaws: {meta_count}")
    print(f"Meta-Reviewer Coverage Ratio:    {ratio:.2f}%")
    print("="*40)

    # Lưu thống kê dạng text
    with open(os.path.join(OUTPUT_DIR, 'summary_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Total Papers: {total}\n")
        f.write(f"Meta Coverage: {ratio:.2f}% ({meta_count}/{total})\n")
        f.write(f"Avg Flaws/Paper: {papers_stats['n_flaws'].mean():.2f}\n")

if __name__ == "__main__":
    main()