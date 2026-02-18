"""
Pipeline Lengkap: Data Generation -> Training -> Evaluation
Script untuk menjalankan seluruh pipeline secara otomatis
"""
import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_data_generation(args):
    """Step 1: Generate dataset dari geomagnetic data"""
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATASET GENERATION")
    logger.info("="*80)
    
    from geomagnetic_dataset_generator import GeomagneticDatasetGenerator
    
    generator = GeomagneticDatasetGenerator(
        event_file=args.event_file,
        station_file=args.station_file,
        output_dir=args.output_dir
    )
    
    if args.demo:
        # Demo mode
        logger.info("Running in DEMO mode")
        demo_date = '2024-01-15'
        demo_station = 'GTO'
        demo_event_info = {'Azm': 135.5, 'Mag': 5.2}
        
        metadata = generator.process_single_event(
            demo_date, demo_station, demo_event_info, args.server_path
        )
        
        if metadata:
            logger.info("Demo completed successfully!")
            return True
        else:
            logger.error("Demo failed!")
            return False
    
    else:
        # Full dataset generation
        metadata_df = generator.generate_dataset_from_events(
            server_path=args.server_path,
            max_events=args.max_events
        )
        
        if metadata_df is not None and len(metadata_df) > 0:
            logger.info(f"Dataset generation completed: {len(metadata_df)} events")
            return True
        else:
            logger.error("Dataset generation failed!")
            return False


def run_training(args, classification_type):
    """Step 2: Train CNN model"""
    logger.info("\n" + "="*80)
    logger.info(f"STEP 2: CNN TRAINING ({classification_type.upper()})")
    logger.info("="*80)
    
    import torch
    from cnn_classifier import (
        GeomagneticSpectrogramDataset, GeomagneticCNN, CNNTrainer
    )
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load metadata
    metadata_path = Path(args.output_dir) / 'metadata' / 'dataset_metadata.csv'
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return False
    
    metadata_df = pd.read_csv(metadata_path)
    logger.info(f"Loaded metadata: {len(metadata_df)} samples")
    
    # Check if we have enough data
    if len(metadata_df) < 10:
        logger.warning(f"Not enough data for training ({len(metadata_df)} samples)")
        return False
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Split data
    label_col = 'azimuth_class' if classification_type == 'azimuth' else 'magnitude_class'
    
    # Filter valid labels
    valid_df = metadata_df[metadata_df[label_col].notna()].copy()
    
    if len(valid_df) < 10:
        logger.warning(f"Not enough labeled data ({len(valid_df)} samples)")
        return False
    
    train_df, val_df = train_test_split(
        valid_df, test_size=0.2, random_state=42, 
        stratify=valid_df[label_col]
    )
    
    # Create datasets
    spectrogram_dir = Path(args.output_dir) / 'spectrograms'
    
    train_dataset = GeomagneticSpectrogramDataset(
        train_df, spectrogram_dir, classification_type, train_transform
    )
    val_dataset = GeomagneticSpectrogramDataset(
        val_df, spectrogram_dir, classification_type, val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=0  # Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=0
    )
    
    # Create model
    num_classes = len(train_dataset.classes)
    model = GeomagneticCNN(num_classes=num_classes, pretrained=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {train_dataset.classes}")
    
    # Train
    trainer = CNNTrainer(model, device=device)
    
    model_save_path = f'best_model_{classification_type}.pth'
    history_save_path = f'training_history_{classification_type}.png'
    
    history = trainer.train(
        train_loader, val_loader, 
        num_epochs=args.epochs, 
        learning_rate=args.learning_rate,
        save_path=model_save_path
    )
    
    # Plot history
    trainer.plot_history(history_save_path)
    
    logger.info(f"Training completed for {classification_type}")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info(f"History plot saved to: {history_save_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline Lengkap: Geomagnetic Dataset Generation + CNN Training'
    )
    
    # Data generation arguments
    parser.add_argument('--event-file', default='intial/event_list.xlsx',
                       help='Path ke file event list')
    parser.add_argument('--station-file', default='intial/lokasi_stasiun.csv',
                       help='Path ke file lokasi stasiun')
    parser.add_argument('--server-path', default='mdata',
                       help='Path ke data server')
    parser.add_argument('--output-dir', default='dataset_spectrogram',
                       help='Output directory')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maksimal jumlah events (None = semua)')
    
    # Pipeline control
    parser.add_argument('--demo', action='store_true',
                       help='Run demo mode (single event)')
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip data generation (use existing dataset)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (only generate data)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size untuk training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Jumlah epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--classification', 
                       choices=['azimuth', 'magnitude', 'both'], 
                       default='both',
                       help='Tipe klasifikasi')
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("\n" + "#"*80)
    logger.info("GEOMAGNETIC DATASET GENERATION & CNN TRAINING PIPELINE")
    logger.info("#"*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration:")
    logger.info(f"  Event file: {args.event_file}")
    logger.info(f"  Station file: {args.station_file}")
    logger.info(f"  Server path: {args.server_path}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Max events: {args.max_events}")
    logger.info(f"  Demo mode: {args.demo}")
    logger.info(f"  Classification: {args.classification}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    
    # Step 1: Data Generation
    if not args.skip_generation:
        success = run_data_generation(args)
        if not success:
            logger.error("Pipeline failed at data generation step")
            return
    else:
        logger.info("Skipping data generation (using existing dataset)")
    
    # Step 2: Training
    if not args.skip_training and not args.demo:
        if args.classification in ['azimuth', 'both']:
            success = run_training(args, 'azimuth')
            if not success:
                logger.warning("Azimuth training failed or skipped")
        
        if args.classification in ['magnitude', 'both']:
            success = run_training(args, 'magnitude')
            if not success:
                logger.warning("Magnitude training failed or skipped")
    else:
        logger.info("Skipping training")
    
    # Summary
    logger.info("\n" + "#"*80)
    logger.info("PIPELINE COMPLETED!")
    logger.info("#"*80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("\nGenerated files:")
    logger.info(f"  - Spectrograms: {args.output_dir}/spectrograms/")
    logger.info(f"  - Metadata: {args.output_dir}/metadata/")
    
    if not args.skip_training and not args.demo:
        logger.info(f"  - Models: best_model_*.pth")
        logger.info(f"  - Training plots: training_history_*.png")


if __name__ == '__main__':
    main()
