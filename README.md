# ZigBee Security Research and Simulation

This project aims to analyze security vulnerabilities in the ZigBee IEEE 802.15.4 protocol through network simulation and machine learning-based intrusion detection.

## Project Overview

The project consists of three main components:
1. Research and analysis of ZigBee protocol vulnerabilities
2. Network simulation using NS-3
3. Machine learning-based intrusion detection system

## Project Structure

```
zigbee-sim/
├── data/                    # Data collection and storage
├── models/                  # Trained neural network models
├── notebooks/              # Jupyter notebooks for analysis
├── src/
│   ├── network/            # NS-3 network simulation code
│   ├── attacks/            # Attack simulation implementations
│   ├── detection/          # Intrusion detection system
│   └── utils/              # Utility functions and helpers
├── tests/                  # Test files
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Dependencies

- Python 3.8+
- NS-3
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib (for visualization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/zigbee-sim.git
cd zigbee-sim
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Network Simulation:
```bash
python src/network/simulate.py
```

2. Attack Simulation:
```bash
python src/attacks/run_attack.py
```

3. Intrusion Detection:
```bash
python src/detection/train.py
python src/detection/detect.py
```