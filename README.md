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

├── simulate_network.py
├── network.py             # Test files
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Dependencies
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
pip install numpy
pip install matplotlib
```

## Usage

1. Network Simulation:
```bash
python simulate_network.py
```
Use a keyboard interrupt to simulate the attack after the regular traffic.

