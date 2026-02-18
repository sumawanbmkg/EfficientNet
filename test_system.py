"""
Test System - Validasi instalasi dan functionality
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test semua dependencies"""
    logger.info("\n" + "="*60)
    logger.info("Testing Dependencies...")
    logger.info("="*60)
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('PIL', 'Pillow'),
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('sklearn', 'Scikit-learn'),
        ('seaborn', 'Seaborn')
    ]
    
    all_ok = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {name:20s} - OK")
        except ImportError:
            logger.error(f"✗ {name:20s} - NOT FOUND")
            all_ok = False
    
    return all_ok


def test_file_structure():
    """Test struktur file dan folder"""
    logger.info("\n" + "="*60)
    logger.info("Testing File Structure...")
    logger.info("="*60)
    
    required_files = [
        'geomagnetic_dataset_generator.py',
        'cnn_classifier.py',
        'run_pipeline.py',
        'visualize_dataset.py',
        'requirements.txt',
        'README_DATASET_GENERATOR.md',
        'QUICK_START.md'
    ]
    
    required_dirs = [
        'intial'
    ]
    
    all_ok = True
    
    # Check files
    for file in required_files:
        if Path(file).exists():
            logger.info(f"✓ {file:40s} - OK")
        else:
            logger.error(f"✗ {file:40s} - NOT FOUND")
            all_ok = False
    
    # Check directories
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            logger.info(f"✓ {dir_name:40s} - OK")
        else:
            logger.error(f"✗ {dir_name:40s} - NOT FOUND")
            all_ok = False
    
    return all_ok


def test_data_files():
    """Test ketersediaan data files"""
    logger.info("\n" + "="*60)
    logger.info("Testing Data Files...")
    logger.info("="*60)
    
    data_files = [
        'intial/event_list.xlsx',
        'intial/lokasi_stasiun.csv',
        'intial/read_mdata.py',
        'intial/signal_processing.py'
    ]
    
    all_ok = True
    
    for file in data_files:
        if Path(file).exists():
            logger.info(f"✓ {file:40s} - OK")
        else:
            logger.warning(f"⚠ {file:40s} - NOT FOUND (optional)")
    
    # Check mdata directory
    if Path('mdata').exists():
        logger.info(f"✓ {'mdata/':40s} - OK")
    else:
        logger.warning(f"⚠ {'mdata/':40s} - NOT FOUND (required for data generation)")
        all_ok = False
    
    return all_ok


def test_module_imports():
    """Test import custom modules"""
    logger.info("\n" + "="*60)
    logger.info("Testing Custom Modules...")
    logger.info("="*60)
    
    modules = [
        ('geomagnetic_dataset_generator', 'GeomagneticDatasetGenerator'),
        ('cnn_classifier', 'GeomagneticCNN'),
        ('visualize_dataset', 'plot_class_distribution')
    ]
    
    all_ok = True
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                logger.info(f"✓ {module_name:40s} - OK")
            else:
                logger.error(f"✗ {module_name:40s} - {class_name} not found")
                all_ok = False
        except Exception as e:
            logger.error(f"✗ {module_name:40s} - Import error: {e}")
            all_ok = False
    
    return all_ok


def test_pytorch_device():
    """Test PyTorch dan CUDA availability"""
    logger.info("\n" + "="*60)
    logger.info("Testing PyTorch Configuration...")
    logger.info("="*60)
    
    try:
        import torch
        
        logger.info(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  Number of GPUs: {torch.cuda.device_count()}")
        else:
            logger.warning("⚠ CUDA not available - will use CPU (slower)")
        
        # Test tensor creation
        x = torch.randn(10, 10)
        logger.info(f"✓ Tensor creation test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ PyTorch test failed: {e}")
        return False


def test_signal_processing():
    """Test signal processing functions"""
    logger.info("\n" + "="*60)
    logger.info("Testing Signal Processing...")
    logger.info("="*60)
    
    try:
        import numpy as np
        from scipy import signal
        
        # Generate test signal
        t = np.linspace(0, 1, 1000)
        test_signal = np.sin(2 * np.pi * 10 * t)
        
        # Test bandpass filter
        nyquist = 500
        low = 5 / nyquist
        high = 15 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, test_signal)
        
        logger.info(f"✓ Bandpass filter test passed")
        
        # Test spectrogram
        f, t, Sxx = signal.spectrogram(test_signal, fs=1000)
        
        logger.info(f"✓ Spectrogram generation test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Signal processing test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "#"*60)
    logger.info("SYSTEM VALIDATION TEST")
    logger.info("#"*60)
    
    results = {
        'Dependencies': test_imports(),
        'File Structure': test_file_structure(),
        'Data Files': test_data_files(),
        'Custom Modules': test_module_imports(),
        'PyTorch': test_pytorch_device(),
        'Signal Processing': test_signal_processing()
    }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:20s}: {status}")
        if not result:
            all_passed = False
    
    logger.info("="*60)
    
    if all_passed:
        logger.info("\n✓ ALL TESTS PASSED!")
        logger.info("System is ready for dataset generation and training.")
        logger.info("\nNext steps:")
        logger.info("  1. Run demo: python run_pipeline.py --demo")
        logger.info("  2. Generate dataset: python geomagnetic_dataset_generator.py")
        logger.info("  3. Train model: python cnn_classifier.py")
        return 0
    else:
        logger.error("\n✗ SOME TESTS FAILED!")
        logger.error("Please fix the issues above before proceeding.")
        logger.error("\nCommon fixes:")
        logger.error("  - Install missing packages: pip install -r requirements.txt")
        logger.error("  - Check file paths and directory structure")
        logger.error("  - Ensure data files are in correct locations")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
