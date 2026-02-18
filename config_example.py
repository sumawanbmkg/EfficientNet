"""
Configuration Example
Contoh konfigurasi untuk customize sistem
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    # Path ke file input
    'event_file': 'intial/event_list.xlsx',
    'station_file': 'intial/lokasi_stasiun.csv',
    'server_path': 'mdata',
    
    # Output directory
    'output_dir': 'dataset_spectrogram',
    
    # Processing limits
    'max_events': None,  # None = process all events
}


# ============================================================================
# SIGNAL PROCESSING CONFIGURATION
# ============================================================================

SIGNAL_CONFIG = {
    # Sampling rate (Hz)
    'sampling_rate': 1.0,  # 1 Hz untuk data 1-detik
    
    # PC3 frequency range (Hz)
    'pc3_low': 0.01,   # 10 mHz
    'pc3_high': 0.045,  # 45 mHz
    
    # Bandpass filter
    'filter_order': 4,  # Butterworth filter order
    
    # Noise removal
    'median_window': 60,  # seconds
    
    # Magnetic storm detection
    'storm_threshold_h': 150,  # nT
    'storm_threshold_d': 100,  # nT
    'storm_threshold_z': 150,  # nT
    'kp_threshold': 4.0,  # Kp index threshold
}


# ============================================================================
# SPECTROGRAM CONFIGURATION
# ============================================================================

SPECTROGRAM_CONFIG = {
    # STFT parameters
    'nperseg': 256,  # Window length
    'noverlap': None,  # None = 50% overlap (nperseg // 2)
    'window': 'hann',  # Window type
    
    # Visualization
    'freq_limit': 100,  # mHz (0-100 mHz display range)
    'colormap': 'viridis',  # Colormap untuk spectrogram
    'dpi': 150,  # Resolution
    'show_colorbar': True,
}


# ============================================================================
# CLASSIFICATION CONFIGURATION
# ============================================================================

CLASSIFICATION_CONFIG = {
    # Azimuth classes (8 directions)
    'azimuth_classes': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
    'azimuth_ranges': {
        'N':  (337.5, 22.5),
        'NE': (22.5, 67.5),
        'E':  (67.5, 112.5),
        'SE': (112.5, 157.5),
        'S':  (157.5, 202.5),
        'SW': (202.5, 247.5),
        'W':  (247.5, 292.5),
        'NW': (292.5, 337.5),
    },
    
    # Magnitude classes (5 categories)
    'magnitude_classes': ['Small', 'Moderate', 'Medium', 'Large', 'Major'],
    'magnitude_ranges': {
        'Small': (0, 4.0),
        'Moderate': (4.0, 5.0),
        'Medium': (5.0, 6.0),
        'Large': (6.0, 7.0),
        'Major': (7.0, 10.0),
    },
}


# ============================================================================
# CNN TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    # Model architecture
    'backbone': 'resnet18',  # resnet18, resnet50, efficientnet_b0
    'pretrained': True,
    'dropout': 0.5,
    
    # Training parameters
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    
    # Data split
    'train_split': 0.8,
    'val_split': 0.2,
    'random_seed': 42,
    
    # Optimizer
    'optimizer': 'adam',  # adam, sgd, adamw
    'momentum': 0.9,  # for SGD
    
    # Learning rate scheduler
    'scheduler': 'plateau',  # plateau, step, cosine
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    
    # Early stopping
    'early_stopping': True,
    'early_stopping_patience': 10,
    
    # Device
    'device': 'auto',  # auto, cuda, cpu
}


# ============================================================================
# DATA AUGMENTATION CONFIGURATION
# ============================================================================

AUGMENTATION_CONFIG = {
    # Image size
    'image_size': (224, 224),
    
    # Training augmentation
    'train_augmentation': {
        'horizontal_flip': True,
        'vertical_flip': False,
        'rotation': 10,  # degrees
        'brightness': 0.1,
        'contrast': 0.1,
    },
    
    # Normalization (ImageNet stats)
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
}


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

VISUALIZATION_CONFIG = {
    # Plot style
    'style': 'whitegrid',  # seaborn style
    'figure_dpi': 150,
    'font_size': 11,
    
    # Colors
    'color_palette': 'Set3',
    'heatmap_cmap': 'YlOrRd',
    
    # Output directory
    'output_dir': 'visualizations',
}


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'log_file': 'pipeline.log',
    'console_output': True,
}


# ============================================================================
# ADVANCED CONFIGURATION
# ============================================================================

ADVANCED_CONFIG = {
    # Parallel processing
    'num_workers': 4,  # for DataLoader
    'multiprocessing': False,
    
    # Memory management
    'pin_memory': True,  # for CUDA
    'persistent_workers': False,
    
    # Checkpointing
    'save_frequency': 5,  # save every N epochs
    'keep_best_only': True,
    
    # Validation
    'validate_every': 1,  # validate every N epochs
    
    # Mixed precision training
    'mixed_precision': False,  # requires CUDA
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(config_name):
    """
    Get configuration by name
    
    Args:
        config_name: Name of configuration
        
    Returns:
        Configuration dictionary
    """
    configs = {
        'data': DATA_CONFIG,
        'signal': SIGNAL_CONFIG,
        'spectrogram': SPECTROGRAM_CONFIG,
        'classification': CLASSIFICATION_CONFIG,
        'training': TRAINING_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'logging': LOGGING_CONFIG,
        'advanced': ADVANCED_CONFIG,
    }
    
    return configs.get(config_name, {})


def print_config(config_name=None):
    """
    Print configuration
    
    Args:
        config_name: Name of configuration (None = print all)
    """
    if config_name:
        config = get_config(config_name)
        print(f"\n{config_name.upper()} CONFIGURATION:")
        print("=" * 60)
        for key, value in config.items():
            print(f"{key:30s}: {value}")
    else:
        # Print all configurations
        configs = [
            'data', 'signal', 'spectrogram', 'classification',
            'training', 'augmentation', 'visualization', 'logging', 'advanced'
        ]
        for name in configs:
            print_config(name)
            print()


def update_config(config_name, updates):
    """
    Update configuration
    
    Args:
        config_name: Name of configuration
        updates: Dictionary with updates
        
    Returns:
        Updated configuration
    """
    config = get_config(config_name)
    config.update(updates)
    return config


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

PRESETS = {
    # Quick test preset
    'quick_test': {
        'data': {'max_events': 10},
        'training': {'num_epochs': 10, 'batch_size': 8},
    },
    
    # High quality preset
    'high_quality': {
        'spectrogram': {'nperseg': 512, 'dpi': 300},
        'training': {'num_epochs': 100, 'batch_size': 32},
    },
    
    # Fast training preset
    'fast_training': {
        'training': {
            'num_epochs': 30,
            'batch_size': 32,
            'learning_rate': 0.01,
        },
    },
    
    # GPU optimized preset
    'gpu_optimized': {
        'training': {
            'batch_size': 64,
            'device': 'cuda',
        },
        'advanced': {
            'num_workers': 8,
            'pin_memory': True,
            'mixed_precision': True,
        },
    },
}


def apply_preset(preset_name):
    """
    Apply preset configuration
    
    Args:
        preset_name: Name of preset
        
    Returns:
        Dictionary with all configurations
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    preset = PRESETS[preset_name]
    
    # Apply updates
    for config_name, updates in preset.items():
        globals()[f"{config_name.upper()}_CONFIG"].update(updates)
    
    print(f"Applied preset: {preset_name}")
    return preset


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    # Example 1: Print all configurations
    print("="*60)
    print("ALL CONFIGURATIONS")
    print("="*60)
    print_config()
    
    # Example 2: Get specific configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    training_config = get_config('training')
    for key, value in training_config.items():
        print(f"{key:30s}: {value}")
    
    # Example 3: Update configuration
    print("\n" + "="*60)
    print("UPDATE CONFIGURATION")
    print("="*60)
    updated = update_config('training', {'num_epochs': 100, 'batch_size': 32})
    print("Updated training config:")
    for key, value in updated.items():
        print(f"{key:30s}: {value}")
    
    # Example 4: Apply preset
    print("\n" + "="*60)
    print("APPLY PRESET")
    print("="*60)
    apply_preset('quick_test')
    print("\nAfter applying 'quick_test' preset:")
    print(f"Max events: {DATA_CONFIG['max_events']}")
    print(f"Num epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
