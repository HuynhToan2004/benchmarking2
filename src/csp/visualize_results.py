# import json
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # --- CẤU HÌNH ---
# # Đường dẫn đến file kết quả JSONL của bạn
# JSONL_FILE = r"D:\Code\Python\Research\LLMs_reviewer\Benchmarking\output\all_results_final.jsonl"
# OUTPUT_IMG_DIR = "analysis_images"

# # Tạo thư mục lưu ảnh
# os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
# sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
# plt.rcParams['figure.figsize'] = (10, 6)

# def load_and_process_data(filepath):
#     """Đọc JSONL và làm phẳng (flatten) dữ liệu để vẽ biểu đồ"""
#     data = []
    
#     print(f"Reading file: {filepath}")
#     with open(filepath, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             try:
#                 paper = json.loads(line)
#             except json.JSONDecodeError:
#                 continue # Bỏ qua dòng lỗi
                
#             paper_id = paper.get('paper_id', f'Unknown_{i}')
            
#             # --- SỬA LỖI Ở ĐÂY: Thử cả 'Decision' và 'decision' ---
#             raw_decision = paper.get('Decision') or paper.get('decision') or 'Unknown'
            
#             decision = 'Unknown'
#             if raw_decision and 'accept' in str(raw_decision).lower():
#                 decision = 'Accept'
#             elif raw_decision and 'reject' in str(raw_decision).lower():
#                 decision = 'Reject'

#             # Lấy Meta Scores
#             meta_cps = None
#             reviews = paper.get('reviews_evaluation', [])
            
#             for rev in reviews:
#                 if rev.get('type') == 'Meta':
#                     # An toàn hơn khi truy cập nested dict
#                     metrics = rev.get('metrics', {})
#                     meta_cps = metrics.get('CPS')
#                     break
            
#             # Duyệt qua từng reviewer
#             for rev in reviews:
#                 metrics = rev.get('metrics', {})
#                 row = {
#                     'Paper ID': paper_id,
#                     'Decision': decision,
#                     'Reviewer Type': rev.get('type', 'Unknown'), 
#                     'Reviewer ID': rev.get('reviewer_id', 'Unknown'),
#                     'NSR': metrics.get('NSR'),
#                     'CPS': metrics.get('CPS'),
#                     'Meta_CPS_Ref': meta_cps
#                 }
#                 data.append(row)
                
#     df = pd.DataFrame(data)
    
#     # Kiểm tra xem có dữ liệu không
#     print(f"Total rows raw: {len(df)}")
    
#     # Loại bỏ Unknown
#     df = df[df['Decision'] != 'Unknown']
#     print(f"Rows after filtering Unknown decision: {len(df)}")
    
#     if df.empty:
#         print("❌ CẢNH BÁO: Không có dữ liệu nào hợp lệ (Decision=Accept/Reject). Kiểm tra lại file JSONL!")
#         return df

#     # Đổi tên hiển thị
#     df['Reviewer Type'] = df['Reviewer Type'].replace({
#         'LLM': 'SEA_LLM',
#         'Human': 'Human Reviewers',
#         'Meta': 'Meta Reviewer'
#     })
    
#     return df

# def plot_metric_by_decision(df, metric, title, filename):
#     """Vẽ Boxplot phân tách theo Accept/Reject"""
#     plt.figure(figsize=(12, 6))
    
#     # Vẽ Boxplot
#     ax = sns.boxplot(
#         data=df, 
#         x="Decision", 
#         y=metric, 
#         hue="Reviewer Type",
#         palette="Set2",
#         showfliers=False # Ẩn outlier quá xa để nhìn rõ box
#     )
    
#     # Vẽ thêm Stripplot (các chấm) để thấy mật độ dữ liệu
#     sns.stripplot(
#         data=df, 
#         x="Decision", 
#         y=metric, 
#         hue="Reviewer Type", 
#         dodge=True, 
#         alpha=0.4, 
#         color='black',
#         ax=ax,
#         legend=False
#     )

#     plt.title(title, fontweight='bold')
#     plt.ylabel(metric)
#     plt.xlabel("Paper Final Decision")
    
#     # Lưu ảnh
#     plt.tight_layout()
#     save_path = os.path.join(OUTPUT_IMG_DIR, filename)
#     plt.savefig(save_path, dpi=300)
#     print(f"Saved: {save_path}")
#     plt.close()

# def plot_alignment_scatter(df):
#     """Vẽ Scatter plot so sánh Human/LLM với Meta Reviewer (Chỉ tính trên tập Reject)"""
#     # Chỉ lấy các bài Reject (nơi cần bắt lỗi)
#     df_reject = df[df['Decision'] == 'Reject'].copy()
    
#     # --- XỬ LÝ HUMAN ---
#     df_human = df_reject[df_reject['Reviewer Type'] == 'Human Reviewers']
    
#     # Group by Paper ID: 
#     # - Tính trung bình CPS của các Human
#     # - Lấy giá trị Meta_CPS_Ref (vì trong cùng 1 paper thì giá trị này giống nhau)
#     # Sử dụng .agg để lấy giá trị đầu tiên của Meta_CPS_Ref (first)
#     if not df_human.empty:
#         df_human_avg = df_human.groupby('Paper ID').agg({
#             'CPS': 'mean',
#             'Meta_CPS_Ref': 'first' 
#         }).reset_index()
#         # Loại bỏ các bài không có Meta Review
#         df_human_avg = df_human_avg.dropna(subset=['Meta_CPS_Ref'])
#     else:
#         df_human_avg = pd.DataFrame(columns=['Paper ID', 'CPS', 'Meta_CPS_Ref'])

#     # --- XỬ LÝ LLM ---
#     df_llm = df_reject[df_reject['Reviewer Type'] == 'SEA_LLM (Ours)']
#     # Chỉ lấy các cột cần thiết và loại bỏ nếu thiếu Meta Review
#     df_llm = df_llm[['Paper ID', 'CPS', 'Meta_CPS_Ref']].dropna(subset=['Meta_CPS_Ref'])
    
#     # --- VẼ BIỂU ĐỒ ---
#     plt.figure(figsize=(8, 8))
    
#     # 1. Vẽ điểm Human (Màu xám)
#     if not df_human_avg.empty:
#         plt.scatter(
#             df_human_avg['Meta_CPS_Ref'], 
#             df_human_avg['CPS'], 
#             alpha=0.5, 
#             label='Avg Human vs. Meta', 
#             color='gray',
#             s=60
#         )
    
#     # 2. Vẽ điểm LLM (Màu xanh)
#     if not df_llm.empty:
#         plt.scatter(
#             df_llm['Meta_CPS_Ref'], 
#             df_llm['CPS'], 
#             alpha=0.7, 
#             label='SEA_LLM vs. Meta', 
#             color='blue',
#             marker='*',
#             s=120
#         )
    
#     # Tìm giới hạn trục để vẽ đường chéo
#     # Xử lý trường hợp không có dữ liệu để tránh lỗi max() arg is an empty sequence
#     all_max_vals = [0]
#     if not df_human_avg.empty:
#         all_max_vals.append(df_human_avg['CPS'].max())
#         all_max_vals.append(df_human_avg['Meta_CPS_Ref'].max())
#     if not df_llm.empty:
#         all_max_vals.append(df_llm['CPS'].max())
#         all_max_vals.append(df_llm['Meta_CPS_Ref'].max())
        
#     max_val = max(all_max_vals) + 2
    
#     # 3. Vẽ đường chéo y=x (Perfect Alignment)
#     plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Alignment (y=x)')
    
#     plt.title("Alignment with Meta-Reviewer (Rejection Cases)", fontweight='bold')
#     plt.xlabel("Meta Reviewer CPS (Ground Truth)")
#     plt.ylabel("Model/Human CPS")
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     plt.tight_layout()
#     save_path = os.path.join(OUTPUT_IMG_DIR, "alignment_scatter.png")
#     plt.savefig(save_path, dpi=300)
#     print(f"Saved: {save_path}")
#     plt.close()

# def print_summary_stats(df):
#     """In bảng thống kê số liệu"""
#     print("\n=== SUMMARY STATISTICS ===")
#     summary = df.groupby(['Decision', 'Reviewer Type'])[['NSR', 'CPS']].agg(['mean', 'std', 'count'])
#     print(summary)
    
#     # Lưu ra CSV để bạn copy vào báo cáo
#     summary.to_csv(os.path.join(OUTPUT_IMG_DIR, "summary_stats.csv"))
#     print("\nStats saved to summary_stats.csv")

# # --- MAIN EXECUTION ---
# if __name__ == "__main__":
#     if not os.path.exists(JSONL_FILE):
#         print(f"❌ Error: File not found at {JSONL_FILE}")
#     else:
#         # 1. Load Data
#         df = load_and_process_data(JSONL_FILE)
        
#         # 2. In thống kê
#         print_summary_stats(df)
        
#         # 3. Vẽ biểu đồ NSR (Chất lượng nội dung - Độ nhiễu)
#         plot_metric_by_decision(
#             df, 
#             metric="NSR", 
#             title="Noise-to-Signal Ratio (Lower is Better)", 
#             filename="nsr_comparison.png"
#         )
        
#         # 4. Vẽ biểu đồ CPS (Độ ưu tiên vấn đề)
#         plot_metric_by_decision(
#             df, 
#             metric="CPS", 
#             title="Critical Priority Score (Higher is Better)", 
#             filename="cps_comparison.png"
#         )
        
#         # 5. Vẽ biểu đồ tương quan (Alignment)
#         plot_alignment_scatter(df)
        
#         print("\n✅ All visualizations generated!")

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CẤU HÌNH ---
JSONL_FILE = r"D:\Code\Python\Research\LLMs_reviewer\Benchmarking\output\all_results_final.jsonl"
OUTPUT_IMG_DIR = "analysis_images"
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Cấu hình giao diện
sns.set_theme(style="white", context="paper", font_scale=1.4)
plt.rcParams['figure.figsize'] = (10, 8)

def load_reject_data(filepath):
    """Chỉ load dữ liệu của các bài bị REJECT để phân tích năng lực bắt lỗi"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                paper = json.loads(line)
                # Lấy decision
                raw_dec = paper.get('Decision') or paper.get('decision') or ''
                if 'reject' not in str(raw_dec).lower():
                    continue

                paper_id = paper.get('paper_id')
                
                for rev in paper.get('reviews_evaluation', []):
                    # Bỏ qua Meta Reviewer trong biểu đồ này để tập trung so sánh AI vs Human
                    if rev['type'] == 'Meta':
                        continue
                        
                    metrics = rev.get('metrics', {})
                    row = {
                        'Paper ID': paper_id,
                        'Reviewer Type': rev.get('type'),
                        'NSR': metrics.get('NSR', 0),
                        'CPS': metrics.get('CPS', 0)
                    }
                    data.append(row)
            except: continue
            
    df = pd.DataFrame(data)
    df['Reviewer Type'] = df['Reviewer Type'].replace({
        'LLM': 'SEA_LLM (Ours)',
        'Human': 'Human Reviewers'
    })
    return df

def plot_quadrant(df):
    plt.figure(figsize=(10, 8))
    
    # 1. Vẽ Scatter Plot
    # Dùng alpha thấp để thấy mật độ, s (size) to để dễ nhìn
    sns.scatterplot(
        data=df, 
        x="NSR", 
        y="CPS", 
        hue="Reviewer Type", 
        style="Reviewer Type",
        palette={"SEA_LLM (Ours)": "blue", "Human Reviewers": "gray"},
        s=100,
        alpha=0.7
    )

    # 2. Vẽ đường phân chia Quadrant (Trung bình vị)
    # Ta dùng ngưỡng NSR=5 (chấp nhận được) và CPS=5 (mức trung bình)
    # Bạn có thể điều chỉnh ngưỡng này tùy theo dữ liệu thực tế
    THRESHOLD_NSR = 5.0 
    THRESHOLD_CPS = 6.0 
    
    plt.axvline(x=THRESHOLD_NSR, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=THRESHOLD_CPS, color='red', linestyle='--', alpha=0.5)

    # 3. Gắn nhãn cho 4 góc (Annotations)
    # Góc Trái-Trên (Tốt nhất)
    plt.text(0.5, 14, "THE IDEAL EXPERT\n(High Priority, Concise)", 
             fontsize=10, color='green', fontweight='bold', ha='left')
    
    # Góc Phải-Trên (AI hay nằm đây)
    plt.text(90, 14, "THE HYPER-CRITICAL\n(Good Priority, Verbose)", 
             fontsize=10, color='blue', fontweight='bold', ha='right')
    
    # Góc Trái-Dưới
    plt.text(0.5, 1, "THE PASSIVE OBSERVER\n(Low Priority, Concise)", 
             fontsize=10, color='gray', ha='left')
             
    # Góc Phải-Dưới (Tệ nhất)
    plt.text(90, 1, "THE DISTRACTED\n(Low Priority, Verbose)", 
             fontsize=10, color='red', ha='right')

    # 4. Giới hạn trục để nhìn rõ (Zoom vào vùng quan trọng)
    # NSR có thể lên tới 100, nhưng phần lớn dữ liệu quan trọng < 20
    # Tuy nhiên để thấy toàn cảnh ta cứ để auto hoặc giới hạn nếu cần
    plt.xlim(-1, 102) 
    plt.ylim(-1, 16)

    plt.title("Reviewer Profiling Matrix (Reject Cases)", fontweight='bold')
    plt.xlabel("Noise-to-Signal Ratio (Lower is Better)")
    plt.ylabel("Critical Priority Score (Higher is Better)")
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_IMG_DIR, "reviewer_quadrant_matrix.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved chart to: {save_path}")
    plt.close()

if __name__ == "__main__":
    df = load_reject_data(JSONL_FILE)
    if not df.empty:
        plot_quadrant(df)
    else:
        print("No data found.")