"""
Neural network training module for ZigBee intrusion detection.
This module handles the training of the neural network for attack detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

class ZigBeeDataset(Dataset):
    """Custom dataset for ZigBee network data."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self) -> int:
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

class IntrusionDetectionModel(nn.Module):
    """Neural network model for intrusion detection."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_classes: int = 2):
        super(IntrusionDetectionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class ModelTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        """
        Initialize the model trainer.
        
        Args:
            model: The neural network model to train
            learning_rate: Learning rate for the optimizer
        """
        self.model = model
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train the model for one epoch.
        
        Args:
            dataloader: DataLoader containing training data
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for features, labels in dataloader:
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
        
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            dataloader: DataLoader containing validation data
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in dataloader:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return total_loss / len(dataloader), correct / total

def train_model(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    input_size: int,
    batch_size: int = 32,
    num_epochs: int = 50
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the intrusion detection model.
    
    Args:
        train_data: Training features
        train_labels: Training labels
        val_data: Validation features
        val_labels: Validation labels
        input_size: Size of input features
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Create datasets and dataloaders
    train_dataset = ZigBeeDataset(train_data, train_labels)
    val_dataset = ZigBeeDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and trainer
    model = IntrusionDetectionModel(input_size)
    trainer = ModelTrainer(model)
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
    return model, history

if __name__ == "__main__":
    # Example usage
    # TODO: Load and preprocess data
    # model, history = train_model(train_data, train_labels, val_data, val_labels, input_size) 