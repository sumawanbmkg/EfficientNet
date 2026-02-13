import json
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_visual_report():
    report_path = 'experiments_v2/hierarchical/validation_report_v2.json'
    output_dir = 'experiments_v2/hierarchical'
    
    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found.")
        return

    with open(report_path, 'r') as f:
        data = json.load(f)

    classes = ['Normal', 'Moderate', 'Medium', 'Large']
    metrics = data['system_hierarchical_metrics']
    
    precision = [metrics[c]['precision'] for c in classes]
    recall = [metrics[c]['recall'] for c in classes]
    f1 = [metrics[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(12, 7))

    rects1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
    rects2 = ax.bar(x, recall, width, label='Recall', color='#e74c3c', alpha=0.8)
    rects3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71', alpha=0.8)

    ax.set_ylabel('Scores (0.0 - 1.0)', fontsize=12)
    ax.set_title('Phase 2.1 Model Evaluation: Visual Metrics by Magnitude Class', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.15)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'vis_performance_metrics.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    # 2. Generate Binary Pie Chart (Normal vs Precursor)
    bin_metrics = data['binary_metrics']
    labels = ['Normal', 'Precursor']
    sizes = [bin_metrics['Normal']['support'], bin_metrics['Precursor']['support']]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = ['#bdc3c7', '#f39c12']
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=(0, 0.1))
    ax2.set_title('Test Set Distribution: Normal vs Precursor Events', fontsize=14)
    
    pie_path = os.path.join(output_dir, 'vis_test_distribution.png')
    plt.savefig(pie_path, dpi=300)
    plt.close()

    print(f"Visual reports generated in {output_dir}")

if __name__ == "__main__":
    generate_visual_report()
