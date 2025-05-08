# ZigBee Security Research and Simulation

This project aims to analyze security vulnerabilities in the ZigBee IEEE 802.15.4 protocol through network simulation.

## Project Overview

The project consists of three main components:
1. Research and analysis of ZigBee protocol vulnerabilities
2. Implementation of a network that simulates a Zigbee network
3. Network condition simulations (regular, attack)

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

## Dependencies

Install dependencies:
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

