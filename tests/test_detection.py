"""
Test module for the intrusion detection system.
"""

import pytest
import numpy as np
import torch
from src.detection.train import IntrusionDetectionModel, ZigBeeDataset
from src.detection.detect import IntrusionDetector
from src.utils.helpers import calculate_metrics

def test_model_initialization():
    """Test that the model initializes correctly."""
    model = IntrusionDetectionModel(input_size=100)
    assert isinstance(model, IntrusionDetectionModel)
    assert len(list(model.parameters())) > 0

def test_dataset_creation():
    """Test that the dataset class works correctly."""
    features = np.random.rand(100, 100)
    labels = np.random.randint(0, 2, size=(100, 1))
    dataset = ZigBeeDataset(features, labels)
    
    assert len(dataset) == 100
    sample_features, sample_labels = dataset[0]
    assert isinstance(sample_features, torch.Tensor)
    assert isinstance(sample_labels, torch.Tensor)

def test_metrics_calculation():
    """Test that metrics are calculated correctly."""
    predictions = np.array([1, 0, 1, 1, 0])
    labels = np.array([1, 0, 0, 1, 0])
    
    metrics = calculate_metrics(predictions, labels)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1

def test_detector_initialization():
    """Test that the detector initializes correctly."""
    # Create a temporary model file
    model = IntrusionDetectionModel(input_size=100)
    torch.save(model.state_dict(), 'temp_model.pt')
    
    detector = IntrusionDetector('temp_model.pt', input_size=100)
    assert isinstance(detector, IntrusionDetector)
    
    # Clean up
    import os
    os.remove('temp_model.pt')

if __name__ == "__main__":
    pytest.main([__file__]) 