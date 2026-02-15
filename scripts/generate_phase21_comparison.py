import matplotlib.pyplot as plt
import numpy as np

# Data dari penelitian Anda
metrics = ['Accuracy', 'Recall\n(Precursor)', 'Recall\n(Normal)', 'Precision', 'F1-Score']
phase_2_1 = [89.0, 93.8, 86.0, 84.2, 88.7]
baseline_vgg16 = [73.2, 81.5, 68.9, 75.1, 78.2]
improvement = [15.8, 12.3, 17.1, 9.1, 10.5]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

# ============================================================================
# SUBPLOT 1: Grouped Bar Chart (Phase 2.1 vs VGG16)
# ============================================================================
x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, phase_2_1, width, label='Phase 2.1 (EfficientNet)', 
                color='#2E7D32', alpha=0.85, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, baseline_vgg16, width, label='Baseline VGG16', 
                color='#C62828', alpha=0.85, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Metrics', fontsize=13, fontweight='bold')
ax1.set_ylabel('Performance (%)', fontsize=13, fontweight='bold')
ax1.set_title('Model Comparison: Phase 2.1 vs Baseline VGG16', 
              fontsize=15, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11)
ax1.legend(fontsize=11, loc='lower right')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add improvement annotations
for i, imp in enumerate(improvement):
    y_pos = max(phase_2_1[i], baseline_vgg16[i]) + 5
    ax1.annotate(f'+{imp:.1f} pp', 
                xy=(i, y_pos), 
                ha='center', va='bottom',
                fontsize=9, color='#1565C0', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3, edgecolor='#1565C0'))

# ============================================================================
# SUBPLOT 2: Improvement Bar Chart
# ============================================================================
colors_improvement = ['#4CAF50' if imp > 10 else '#FFC107' for imp in improvement]
bars3 = ax2.barh(metrics, improvement, color=colors_improvement, 
                 alpha=0.85, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Improvement (Percentage Points)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Metrics', fontsize=13, fontweight='bold')
ax2.set_title('Performance Improvement\n(Phase 2.1 over VGG16)', 
              fontsize=15, fontweight='bold', pad=20)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_xlim(0, max(improvement) * 1.2)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars3, improvement)):
    width = bar.get_width()
    ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
            f'+{val:.1f} pp',
            ha='left', va='center', fontsize=11, fontweight='bold')

# Add average improvement line
avg_improvement = np.mean(improvement)
ax2.axvline(x=avg_improvement, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Average: +{avg_improvement:.1f} pp')
ax2.legend(fontsize=10, loc='lower right')

# Overall title
fig.suptitle('Earthquake Precursor Detection: Phase 2.1 Performance Analysis', 
             fontsize=17, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('d:/multi/ViT/phase21_vs_vgg16_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Grafik tersimpan: d:/multi/ViT/phase21_vs_vgg16_comparison.png")

# ============================================================================
# BONUS: Create Summary Table Visualization
# ============================================================================
fig2, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = [
    ['Metric', 'Phase 2.1', 'Baseline VGG16', 'Improvement'],
    ['Accuracy', '89.0%', '73.2%', '+15.8 pp'],
    ['Recall (Precursor)', '93.8%', '81.5%', '+12.3 pp'],
    ['Recall (Normal)', '86.0%', '68.9%', '+17.1 pp'],
    ['Precision', '84.2%', '75.1%', '+9.1 pp'],
    ['F1-Score', '88.7%', '78.2%', '+10.5 pp']
]

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.2, 0.25, 0.25])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Header styling
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#1976D2')
    cell.set_text_props(weight='bold', color='white')

# Color code improvements
for i in range(1, 6):
    # Phase 2.1 cells (green)
    table[(i, 1)].set_facecolor('#C8E6C9')
    # VGG16 cells (red)
    table[(i, 2)].set_facecolor('#FFCDD2')
    # Improvement cells (yellow)
    table[(i, 3)].set_facecolor('#FFF9C4')
    table[(i, 3)].set_text_props(weight='bold', color='#1565C0')

# Add title
plt.title('Performance Metrics Summary Table', fontsize=15, fontweight='bold', pad=20)

plt.savefig('d:/multi/ViT/phase21_summary_table.png', dpi=150, bbox_inches='tight')
print("✅ Tabel tersimpan: d:/multi/ViT/phase21_summary_table.png")

plt.show()

# Print summary
print("\n" + "="*60)
print("📊 PERFORMANCE SUMMARY")
print("="*60)
print(f"Average Improvement: +{avg_improvement:.1f} percentage points")
print(f"Best Metric: Recall (Normal) with +{max(improvement):.1f} pp improvement")
print(f"Phase 2.1 Average Performance: {np.mean(phase_2_1):.1f}%")
print(f"VGG16 Baseline Average: {np.mean(baseline_vgg16):.1f}%")
print("="*60)
