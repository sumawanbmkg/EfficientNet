"""
Test training script dengan dataset yang ada
Quick test dengan 5 epochs untuk validasi
"""

from train_multi_task import EarthquakeTrainer

# Configuration untuk testing
config = {
    # Dataset
    'dataset_dir': 'dataset_spectrogram_ssh_v21',  # Use v21 dataset (224x224 RGB, no axis)
    'batch_size': 8,  # Small batch for testing
    'val_split': 0.2,
    'test_split': 0.1,
    'num_workers': 0,
    'seed': 42,
    
    # Model
    'backbone': 'resnet18',  # Smaller model for faster testing
    'pretrained': True,
    'num_magnitude_classes': 5,
    'num_azimuth_classes': 8,
    'dropout_rate': 0.5,
    'learn_weights': True,
    
    # Training (reduced for testing)
    'epochs': 5,  # Just 5 epochs for testing
    'learning_rate': 1e-4,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'scheduler': 'plateau',
}

print("="*80)
print("TESTING TRAINING PIPELINE")
print("="*80)
print(f"Dataset: {config['dataset_dir']}")
print(f"Model: {config['backbone']}")
print(f"Epochs: {config['epochs']} (testing only)")
print(f"Batch size: {config['batch_size']}")
print("="*80)

# Create trainer
trainer = EarthquakeTrainer(config, output_dir='test_experiments')

# Train
print("\nStarting training test...")
history = trainer.train()

print("\n" + "="*80)
print("TRAINING TEST COMPLETED!")
print("="*80)
print(f"Best epoch: {trainer.best_epoch + 1}")
print(f"Best val loss: {trainer.best_val_loss:.4f}")
print(f"Final train mag acc: {history['train_mag_acc'][-1]:.4f}")
print(f"Final train az acc: {history['train_az_acc'][-1]:.4f}")
print(f"Final val mag acc: {history['val_mag_acc'][-1]:.4f}")
print(f"Final val az acc: {history['val_az_acc'][-1]:.4f}")
print(f"\nResults saved to: {trainer.exp_dir}")
print("="*80)
