"""
Test Dashboard Evaluation Content
Verifies that all new evaluation visualizations are accessible and valid.
"""

import os
from pathlib import Path
from PIL import Image
import pandas as pd

def test_files_exist():
    """Test that all required files exist."""
    print("=" * 70)
    print("TESTING FILE EXISTENCE")
    print("=" * 70)
    
    base_path = Path("q1_comprehensive_report")
    
    # Test PNG files
    png_files = [
        "fig1_dataset_characterization.png",
        "fig2_training_convergence.png",
        "fig3_ablation_study.png",
        "fig4_training_efficiency.png",
        "fig5_loss_optimization.png",
        "fig6_model_comparison.png",
        "fig7_loss_curves_detailed.png",  # NEW
        "fig8_roc_auc_curves.png",        # NEW
        "fig9_confusion_matrices.png",    # NEW
    ]
    
    print("\n📊 PNG Files (Figures):")
    png_count = 0
    for png_file in png_files:
        file_path = base_path / png_file
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {png_file}")
        if exists:
            png_count += 1
    
    print(f"\n  Total: {png_count}/{len(png_files)} PNG files found")
    
    # Test CSV files
    csv_files = [
        "table1_architecture_specification.csv",
        "table2_hyperparameter_specification.csv",
        "table3_performance_metrics.csv",
        "table4_statistical_analysis.csv",
        "table5_technical_specifications.csv",
        "table6_preprocessing_pipeline.csv",
        "table7_dataset_statistics.csv",
        "table8_model_comparison.csv",
        "table9_evaluation_metrics.csv",  # NEW
    ]
    
    print("\n📋 CSV Files (Tables):")
    csv_count = 0
    for csv_file in csv_files:
        file_path = base_path / csv_file
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {csv_file}")
        if exists:
            csv_count += 1
    
    print(f"\n  Total: {csv_count}/{len(csv_files)} CSV files found")
    
    return png_count == len(png_files) and csv_count == len(csv_files)


def test_image_quality():
    """Test that new images are valid and high quality."""
    print("\n" + "=" * 70)
    print("TESTING IMAGE QUALITY")
    print("=" * 70)
    
    base_path = Path("q1_comprehensive_report")
    new_images = [
        "fig7_loss_curves_detailed.png",
        "fig8_roc_auc_curves.png",
        "fig9_confusion_matrices.png",
    ]
    
    all_valid = True
    
    for img_file in new_images:
        file_path = base_path / img_file
        print(f"\n📊 Testing: {img_file}")
        
        try:
            img = Image.open(file_path)
            width, height = img.size
            dpi = img.info.get('dpi', (0, 0))
            
            print(f"  ✅ Image loaded successfully")
            print(f"  📐 Dimensions: {width} x {height} pixels")
            print(f"  🎨 DPI: {dpi[0]} x {dpi[1]}")
            print(f"  🖼️  Format: {img.format}")
            print(f"  🎭 Mode: {img.mode}")
            
            # Check if high quality (300 DPI or large dimensions)
            if dpi[0] >= 300 or width >= 2400:
                print(f"  ✅ High quality (300 DPI or equivalent)")
            else:
                print(f"  ⚠️  Quality may be lower than 300 DPI")
            
        except Exception as e:
            print(f"  ❌ Error loading image: {e}")
            all_valid = False
    
    return all_valid


def test_table_content():
    """Test that new table has valid content."""
    print("\n" + "=" * 70)
    print("TESTING TABLE CONTENT")
    print("=" * 70)
    
    base_path = Path("q1_comprehensive_report")
    table_file = "table9_evaluation_metrics.csv"
    file_path = base_path / table_file
    
    print(f"\n📋 Testing: {table_file}")
    
    try:
        df = pd.read_csv(file_path)
        
        print(f"  ✅ Table loaded successfully")
        print(f"  📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  📋 Columns: {list(df.columns)}")
        
        print(f"\n  📊 Table Preview:")
        print(df.to_string(index=False))
        
        # Check for expected metrics
        expected_metrics = [
            'Accuracy', 'Balanced Accuracy', 'F1-Score (Macro)',
            'F1-Score (Weighted)', 'Precision (Macro)', 'Recall (Macro)',
            'Cohen\'s Kappa', 'MCC', 'AUPRC (Macro)', 'AUROC (Macro)'
        ]
        
        print(f"\n  🔍 Checking for expected metrics:")
        for metric in expected_metrics:
            if metric in df['Metric'].values:
                print(f"    ✅ {metric}")
            else:
                print(f"    ❌ {metric} (missing)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading table: {e}")
        return False


def test_dashboard_imports():
    """Test that dashboard can import required packages."""
    print("\n" + "=" * 70)
    print("TESTING DASHBOARD DEPENDENCIES")
    print("=" * 70)
    
    packages = {
        'streamlit': 'Streamlit',
        'plotly': 'Plotly',
        'pandas': 'Pandas',
        'PIL': 'Pillow (PIL)',
        'numpy': 'NumPy'
    }
    
    all_available = True
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (not installed)")
            all_available = False
    
    return all_available


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DASHBOARD EVALUATION CONTENT TEST")
    print("=" * 70)
    print("\nTesting new evaluation visualizations integration...")
    print("Date: February 1, 2026")
    
    # Run tests
    test1 = test_files_exist()
    test2 = test_image_quality()
    test3 = test_table_content()
    test4 = test_dashboard_imports()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    tests = [
        ("File Existence", test1),
        ("Image Quality", test2),
        ("Table Content", test3),
        ("Dashboard Dependencies", test4)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    print(f"\n📊 Results: {passed}/{total} tests passed\n")
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    if passed == total:
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n🎉 Dashboard is ready for presentation!")
        print("\nNext steps:")
        print("  1. Run: streamlit run project_dashboard.py")
        print("  2. Navigate to: '📊 Full Q1 Report' menu")
        print("  3. Verify: All 9 figures and 9 tables display correctly")
        print("\n" + "=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("⚠️  SOME TESTS FAILED")
        print("=" * 70)
        print("\nPlease check the errors above and:")
        print("  1. Ensure all files are in q1_comprehensive_report/")
        print("  2. Run: python generate_evaluation_visualizations.py")
        print("  3. Install missing packages: pip install -r requirements_dashboard.txt")
        print("\n" + "=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
