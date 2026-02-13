import json
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_adv_visual_reports():
    report_path = 'experiments_v2/hierarchical/validation_report_v2.json'
    output_dir = 'experiments_v2/hierarchical'
    
    with open(report_path, 'r') as f:
        data = json.load(f)

    # 1. Comparison Chart: Phase 2.1 vs Champion Q1
    q1_metrics = {
        'Recall Large': 0.70,
        'Precision Large': 0.50,
        'F1-Score Large': 0.58,
        'Overall Acc': 0.75
    }
    
    p2_metrics = {
        'Recall Large': data['system_hierarchical_metrics']['Large']['recall'],
        'Precision Large': data['system_hierarchical_metrics']['Large']['precision'],
        'F1-Score Large': data['system_hierarchical_metrics']['Large']['f1-score'],
        'Overall Acc': data['system_hierarchical_metrics']['accuracy']
    }

    labels = list(q1_metrics.keys())
    q1_values = [q1_metrics[l] for l in labels]
    p2_values = [p2_metrics[l] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.style.use('seaborn-v0_8-muted')
    fig, ax = plt.subplots(figsize=(11, 7))
    
    ax.bar(x - width/2, q1_values, width, label='Champion Q1 (Baseline)', color='#95a5a6')
    ax.bar(x + width/2, p2_values, width, label='Phase 2.1 (Proposed)', color='#2ecc71')

    ax.set_ylabel('Score')
    ax.set_title('Benchmark: Phase 2.1 vs Champion Model Q1', fontsize=15, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    for i, v in enumerate(p2_values):
        ax.text(i + width/2, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vis_comparison_q1.png'), dpi=300)
    plt.close()

    # 2. Radar Chart for Per-Class F1-Score
    classes = ['Normal', 'Moderate', 'Medium', 'Large']
    f1_scores = [data['system_hierarchical_metrics'][c]['f1-score'] for c in classes]
    
    # Radar chart needs to close the loop
    labels_radar = classes
    stats = f1_scores + [f1_scores[0]]
    angles = np.linspace(0, 2*np.pi, len(labels_radar), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='teal', alpha=0.25)
    ax.plot(angles, stats, color='teal', linewidth=2)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, fontsize=12)
    ax.set_title('Model Sensitivity Radar (F1-Score)', size=15, color='teal', y=1.1)
    
    plt.savefig(os.path.join(output_dir, 'vis_radar_performance.png'), dpi=300)
    plt.close()

    print(f"Advanced visual reports generated in {output_dir}")

if __name__ == "__main__":
    generate_adv_visual_reports()
