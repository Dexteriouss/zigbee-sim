"""
Utility functions for the ZigBee security research project.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

def save_data(data: Dict, filename: str) -> None:
    """
    Save data to a file.
    
    Args:
        data: Data to save
        filename: Name of the file to save to
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_data(filename: str) -> Dict:
    """
    Load data from a file.
    
    Args:
        filename: Name of the file to load from
        
    Returns:
        Loaded data
    """
    with open(filename, 'r') as f:
        return json.load(f)

def preprocess_network_data(raw_data: np.ndarray) -> np.ndarray:
    """
    Preprocess raw network data for analysis.
    
    Args:
        raw_data: Raw network data
        
    Returns:
        Preprocessed data
    """
    # TODO: Implement data preprocessing
    return raw_data

def extract_features(data: np.ndarray) -> np.ndarray:
    """
    Extract relevant features from network data.
    
    Args:
        data: Network data
        
    Returns:
        Extracted features
    """
    # TODO: Implement feature extraction
    return data

def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for the model.
    
    Args:
        predictions: Model predictions
        labels: True labels
        
    Returns:
        Dictionary of metrics
    """
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    true_negatives = np.sum((predictions == 0) & (labels == 0))
    
    accuracy = (true_positives + true_negatives) / len(predictions)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def generate_timestamp() -> str:
    """
    Generate a timestamp string.
    
    Returns:
        Current timestamp as string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_experiment_directory(base_dir: str = "experiments") -> str:
    """
    Create a directory for experiment results.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to the created directory
    """
    timestamp = generate_timestamp()
    experiment_dir = os.path.join(base_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

if __name__ == "__main__":
    # Example usage
    data = {"test": 123}
    save_data(data, "data/test.json")
    loaded_data = load_data("data/test.json")
    print(loaded_data) 