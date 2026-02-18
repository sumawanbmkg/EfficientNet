"""
CNN Classifier untuk Geomagnetic Spectrogram
Multi-class classification berdasarkan Azimuth dan Magnitude
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeomagneticSpectrogramDataset(Dataset):
    """Dataset untuk spectrogram geomagnetic"""
    
    def __init__(self, metadata_df, spectrogram_dir, 
                 classification_type='azimuth', transform=None):
        """
        Args:
            metadata_df: DataFrame dengan metadata
            spectrogram_dir: Directory dengan file spectrogram
            classification_type: 'azimuth' atau 'magnitude'
            transform: transformasi untuk image
        """
        self.metadata = metadata_df
        self.spectrogram_dir = Path(spectrogram_dir)
        self.classification_type = classification_type
        self.transform = transform
        
        # Setup class labels
        if classification_type == 'azimuth':
            self.classes = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            self.label_column = 'azimuth_class'
        elif classification_type == 'magnitude':
            self.classes = ['Small', 'Moderate', 'Medium', 'Large', 'Major']
            self.label_column = 'magnitude_class'
        else:
            raise ValueError(f"Unknown classification type: {classification_type}")
        
        # Filter data yang punya label
        self.metadata = self.metadata[
            self.metadata[self.label_column].isin(self.classes)
        ].reset_index(drop=True)
        
        # Create label mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        logger.info(f"Dataset initialized: {len(self.metadata)} samples")
        logger.info(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get metadata
        row = self.metadata.iloc[idx]
        
        # Load 3-component spectrogram image
        img_filename = row['spectrogram_files']['3comp']
        img_path = self.spectrogram_dir / img_filename
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label_str = row[self.label_column]
        label = self.class_to_idx[label_str]
        
        return image, label


class GeomagneticCNN(nn.Module):
    """CNN model untuk klasifikasi spectrogram"""
    
    def __init__(self, num_classes, pretrained=True):
        super(GeomagneticCNN, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class CNNTrainer:
    """Trainer untuk CNN model"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'train_acc': [], 
                       'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, dataloader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=50, 
             learning_rate=0.001, save_path='best_model.pth'):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_acc = 0.0
        
        logger.info("Starting training...")
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"  Best model saved! Val Acc: {val_acc:.2f}%")
        
        logger.info(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%")
        
        return self.history
    
    def plot_history(self, save_path='training_history.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved to {save_path}")


def main():
    """Main function untuk training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN untuk Geomagnetic Classification')
    parser.add_argument('--metadata', default='dataset_spectrogram/metadata/dataset_metadata.csv',
                       help='Path ke metadata CSV')
    parser.add_argument('--spectrogram-dir', default='dataset_spectrogram/spectrograms',
                       help='Directory dengan spectrogram images')
    parser.add_argument('--classification', choices=['azimuth', 'magnitude'], 
                       default='azimuth', help='Tipe klasifikasi')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Load metadata
    metadata_df = pd.read_csv(args.metadata)
    logger.info(f"Loaded metadata: {len(metadata_df)} samples")
    
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
    train_df, val_df = train_test_split(metadata_df, test_size=0.2, 
                                        random_state=42, stratify=metadata_df['azimuth_class'])
    
    # Create datasets
    train_dataset = GeomagneticSpectrogramDataset(
        train_df, args.spectrogram_dir, args.classification, train_transform
    )
    val_dataset = GeomagneticSpectrogramDataset(
        val_df, args.spectrogram_dir, args.classification, val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    
    # Create model
    num_classes = len(train_dataset.classes)
    model = GeomagneticCNN(num_classes=num_classes, pretrained=True)
    
    logger.info(f"Model created: {num_classes} classes")
    logger.info(f"Device: {args.device}")
    
    # Train
    trainer = CNNTrainer(model, device=args.device)
    history = trainer.train(train_loader, val_loader, 
                           num_epochs=args.epochs, 
                           learning_rate=args.lr,
                           save_path=f'best_model_{args.classification}.pth')
    
    # Plot history
    trainer.plot_history(f'training_history_{args.classification}.png')


if __name__ == '__main__':
    main()
