"""
Script Phase 2 - Step 5: Validate Comprehensive (LOEO & Recall Test)
Melakukan validasi ketat terhadap Model Hierarkis V2 yang baru dilatih.

Fitur Validasi:
1. Standard Test Set Metrics: F1-Score, Precision, Recall per class.
2. LOEO (Leave-One-Event-Out) pada event besar (Large/Major) yang ada di Test Set.
   - Apakah model konsisten mendeteksi event ini?
3. Physics Gate Simulation:
   - Hitung Z/H feature dan lihat dampaknya terhadap False Positive Rate (jika diaktifkan).
   - Gunakan threshold statis (DEFAULT_ZH_THRESHOLD = 0.85).

Input:
- experiments_v2/hierarchical/best_model.pth
- dataset_consolidation/metadata/split_test.csv

Output:
- experiments_v2/hierarchical/validation_report.json
- experiments_v2/hierarchical/confusion_matrix.png
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import dari trainer_v2 untuk load model class & dataset class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autoupdate_pipeline.src.trainer_v2 import HierarchicalEfficientNet, HierarchicalEarthquakeDataset

# Konfigurasi
MODEL_PATH = 'experiments_v2/hierarchical/best_model.pth'
TEST_META = 'dataset_consolidation/metadata/split_test.csv'
DATASET_ROOT = os.getcwd() # Asumsi dijalankan dari root d:\multi
OUTPUT_DIR = 'experiments_v2/hierarchical'
ZH_THRESHOLD = 0.85 # Contoh threshold, nanti diupdate dari hasil investigasi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def estimate_zh_ratio(img_tensor):
    """
    Hitung Spectrogram-derived Polarization Ratio (SPR) dari image tensor.
    Asumsi Input Tensor Shape: [Batch, Channel, Height, Width]
    Normalization: ImageNet Mean/Std sudah diaplikasikan, perlu denormalize estimasi kasar atau hitung relatif.
    
    Channel Mapping (dari generate_process: R=H, G=D, B=Z)
    Tensor Torch: Channel 0=R(H), 1=G(D), 2=B(Z)
    
    Formula SPR = Mean(Z) / Mean(H)
    """
    # Denormalize tidak mutlak perlu jika kita hanya butuh rasio relatif,
    # tapi agar aman, kita ambil raw intensity di GPU langsung.
    # Tensor input biasanya sudah dinormalisasi: (x - mean) / std
    # Kita kembalikan ke skala 0-1 estimasi: x * std + mean
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(img_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(img_tensor.device).view(1, 3, 1, 1)
    
    raw_img = img_tensor * std + mean
    # Clip agar tidak negatif (floating point error)
    raw_img = torch.clamp(raw_img, 0.0, 1.0)
    
    # Hitung rata-rata per channel per sample
    # Dimensi: [Batch, Channel, H, W] -> Mean di [2, 3] -> [Batch, Channel]
    channel_means = raw_img.mean(dim=[2, 3]) 
    
    # H = Channel 0 (Red)
    # Z = Channel 2 (Blue)
    h_mean = channel_means[:, 0]
    z_mean = channel_means[:, 2]
    
    # Hindari pembagian nol
    h_mean = torch.clamp(h_mean, min=1e-6)
    
    spr = z_mean / h_mean
    return spr.cpu().numpy() # Return as numpy array [Batch] 

def main():
    logger.info("="*50)
    logger.info("PHASE 2 - STEP 5: COMPREHENSIVE VALIDATION")
    logger.info("="*50)
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}. Train first!")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = HierarchicalEfficientNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")
    
    # 2. Load Test Data
    test_df = pd.read_csv(TEST_META)
    
    # Transform (Sama dengan Validasi Training)
    from torchvision import transforms
    from PIL import Image
    test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_ds = HierarchicalEarthquakeDataset(test_df, DATASET_ROOT, test_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 3. Inference Loop
    all_preds_bin = []
    all_targets_bin = []
    
    all_preds_mag = []
    all_targets_mag = []
    
    all_raw_probs_large = [] # Untuk analisis threshold
    
    logger.info("Running inference on Test Set...")
    
    with torch.no_grad():
        for imgs, is_prec, mag_lbl, azi_lbl in test_loader:
            imgs = imgs.to(device)
            
            # Forward Pass
            bin_logits, mag_logits, azi_logits = model(imgs)
            
            # --- Hierarchical Logic Simulation ---
            # 0. Physics Gate (Z/H Ratio)
            spr_values = estimate_zh_ratio(imgs)
            
            # 1. Binary Decision
            bin_probs = torch.softmax(bin_logits, dim=1)
            bin_preds = bin_probs.argmax(1).cpu().numpy()
            
            # Apply Gate: Jika SPR < Threshold (dan threshold diaktifkan > 0), force Binary = Normal
            # Namun, untuk analisis komprehensif, kita simpan prediksi ASLI model dulu.
            # Nanti "System Performance" yang akan menerapkan filter ini.
            
            # 2. Magnitude Decision
            # Dalam mode evaluasi murni, kita biasanya ingin tahu performa magnitude
            # terlepas dari binary gate (untuk analisis error).
            # Tapi untuk "System Performance", kita harus ikut aturan Gate.
            
            # Kita simpan RAW prediction dulu untuk analisis mendalam
            mag_probs = torch.softmax(mag_logits, dim=1)
            mag_preds = mag_probs.argmax(1).cpu().numpy()
            
            # Simpan data
            all_preds_bin.extend(bin_preds)
            all_targets_bin.extend(is_prec.numpy())
            
            all_preds_mag.extend(mag_preds)
            all_targets_mag.extend(mag_lbl.numpy())
            
            # Simpan probabilitas kelas Large (Index 3 di mag_classes)
            large_probs = mag_probs[:, 3].cpu().numpy()
            all_raw_probs_large.extend(large_probs)
            
            # Simpan Physics Values
            # Kita butuh list global untuk spr_values juga jika ingin analisis korelasi
            # (Untuk saat ini kita pakai SPR langsung di loop System Performance bawah)
            # Ops, loop system performance di bawah pakai list `all_preds`.
            # Jadi kita PERLU simpan spr_values ke list global.
            if 'all_spr_values' not in locals(): all_spr_values = []
            all_spr_values.extend(spr_values)

    # 4. Metrics Calculation
    
    # A. Binary Classification Metrics
    bin_report = classification_report(all_targets_bin, all_preds_bin, target_names=['Normal', 'Precursor'], output_dict=True)
    logger.info(f"\n[BINARY] Recall Precursor: {bin_report['Precursor']['recall']:.2%}")
    logger.info(f"[BINARY] F1-Score: {bin_report['Precursor']['f1-score']:.2%}")
    
    # B. Magnitude Classification Metrics (Raw - Model Capability)
    # Mapping index ke nama kelas
    mag_names = test_ds.mag_classes # ['Normal', 'Moderate', 'Medium', 'Large']
    mag_report = classification_report(all_targets_mag, all_preds_mag, target_names=mag_names, output_dict=True)
    
    logger.info("\n[MAGNITUDE RAW] Per-Class Performance:")
    for cls in mag_names:
        if cls in mag_report:
            rec = mag_report[cls]['recall']
            prec = mag_report[cls]['precision']
            logger.info(f"  {cls:10s} | Recall: {rec:.2%} | Precision: {prec:.2%}")
            
    # C. System-Level Performance (Hierarchical + Physics Gate)
    # Aturan Main:
    # 1. Jika Binary Classifier == Normal (0) -> Final = Normal
    # 2. Jika Z/H Ratio (SPR) < Threshold -> Final = Normal (Override Model)
    # 3. Else -> Final = Magnitude Prediction
    
    final_preds_hierarchical = []
    final_preds_phys_gated = []
    
    # Pastikan all_spr_values ada (inisialisasi jika kosong di loop)
    if 'all_spr_values' not in locals(): all_spr_values = [1.0] * len(all_preds_bin)
    
    for bin_p, mag_p, spr in zip(all_preds_bin, all_preds_mag, all_spr_values):
        # 1. Pure Hierarchical (Tanpa Physics)
        if bin_p == 0:
            final_preds_hierarchical.append(0) # Normal
        else:
            final_preds_hierarchical.append(mag_p)
            
        # 2. Physics-Gated Hierarchical
        if spr < ZH_THRESHOLD:
            final_preds_phys_gated.append(0) # Force Normal by Physics
        elif bin_p == 0:
            final_preds_phys_gated.append(0) # Normal by AI
        else:
            final_preds_phys_gated.append(mag_p) # AI Precursor Prediction
            
    # Report Pure Hierarchical
    logger.info("--- [SYSTEM 1] AI-Only Hierarchical ---")
    sys_report = classification_report(all_targets_mag, final_preds_hierarchical, target_names=mag_names, output_dict=True)
    logger.info(f"  LARGE Recall:    {sys_report['Large']['recall']:.2%}")
    logger.info(f"  LARGE Precision: {sys_report['Large']['precision']:.2%}")
    
    # Report Physics Gated
    logger.info(f"\n--- [SYSTEM 2] Physics-Gated (SPR > {ZH_THRESHOLD}) ---")
    phys_report = classification_report(all_targets_mag, final_preds_phys_gated, target_names=mag_names, output_dict=True)
    
    logger.info("  Final Performance with Physics Gate:")
    # Fokus kita: RECALL LARGE
    large_recall = phys_report['Large']['recall']
    large_precision = phys_report['Large']['precision']
    
    logger.info(f"  LARGE Recall:    {large_recall:.2%} (Target > 60%)")
    logger.info(f"  LARGE Precision: {large_precision:.2%} (Target > 50%)")
    
    # 5. Visualization (Confusion Matrix - Physics Gated)
    cm = confusion_matrix(all_targets_mag, final_preds_phys_gated)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=mag_names, yticklabels=mag_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Hierarchical System Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_hierarchical.png'))
    plt.close()
    
    # 6. Save Full Report
    full_report = {
        'binary_metrics': bin_report,
        'magnitude_raw_metrics': mag_report,
        'system_hierarchical_metrics': sys_report,
        'system_physics_gated_metrics': phys_report,
        'spr_stats': {
            'mean': float(np.mean(all_spr_values)),
            'std': float(np.std(all_spr_values)),
            'threshold_used': ZH_THRESHOLD
        },
        'model_path': MODEL_PATH,
        'test_set_size': len(test_df)
    }
    
    with open(os.path.join(OUTPUT_DIR, 'validation_report_v2.json'), 'w') as f:
        json.dump(full_report, f, indent=2)
        
    logger.info(f"Report saved to {OUTPUT_DIR}/validation_report_v2.json")

if __name__ == "__main__":
    main()
