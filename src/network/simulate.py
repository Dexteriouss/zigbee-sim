"""
Custom ZigBee network simulation module.
This module implements a ZigBee network simulation from scratch.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import random
from dataclasses import dataclass
from enum import Enum

# Types of ZigBee devices
class DeviceType(Enum):
    COORDINATOR = "coordinator"
    ROUTER = "router"
    END_DEVICE = "end_device"

# Represents a ZigBee device in the network
@dataclass
class Device:
    id: int
    type: DeviceType
    x: float
    y: float
    z: float
    tx_power: float
    rx_sensitivity: float
    energy: float
    neighbors: List[int]
    packets_sent: int = 0
    packets_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0

# Initialize a ZigBee network simulation
class ZigBeeNetwork:
    def __init__(self, num_devices: int = 10, area_size: float = 100.0):
        self.num_devices = num_devices
        self.area_size = area_size
        self.devices: List[Device] = []
        self.simulation_time = 0.0
        self.packet_queue: List[Dict] = []
        
    # Set up the initial network configuration
    def setup_network(self) -> None:
        # Create devices
        for i in range(self.num_devices):
            # Determine device type
            if i == 0:
                device_type = DeviceType.COORDINATOR # One coordinator
            elif i < self.num_devices // 2: 
                device_type = DeviceType.ROUTER # Half routers
            else:
                device_type = DeviceType.END_DEVICE # Half end devices
                
            # Random position in the area
            x = random.uniform(0, self.area_size)
            y = random.uniform(0, self.area_size)
            z = 0.0  # 2D simulation
            
            # Create device with initial parameters
            device = Device(
                id=i,
                type=device_type,
                x=x,
                y=y,
                z=z,
                tx_power=0.0,  # dBm
                rx_sensitivity=-85.0,  # dBm
                energy=100.0,  # Initial energy in Joules
                neighbors=[]
            )
            self.devices.append(device)
            
        # Build neighbor lists based on distance
        self._build_neighbor_lists()
        
    # Build neighbor lists for each device based on distance
    def _build_neighbor_lists(self) -> None:
        for i, device1 in enumerate(self.devices):
            for j, device2 in enumerate(self.devices):
                if i != j:
                    distance = self._calculate_distance(device1, device2)
                    # Assume communication range of 30 meters
                    if distance <= 30.0:
                        device1.neighbors.append(j)
                        
    # Calculate distance between two devices
    def _calculate_distance(self, device1: Device, device2: Device) -> float:
        return np.sqrt(
            (device1.x - device2.x)**2 +
            (device1.y - device2.y)**2 +
            (device1.z - device2.z)**2
        )
        
    # Calculate path loss using log-distance path loss model
    def _calculate_path_loss(self, distance: float) -> float:
        # Free space path loss exponent (typically 2-4)
        path_loss_exponent = 2.0
        # Reference distance (1 meter)
        d0 = 1.0
        # Path loss at reference distance (dB)
        PL0 = 40.0
        
        return PL0 + 10 * path_loss_exponent * np.log10(distance / d0)
        
    # Determine if two devices can communicate based on distance and power
    def _can_communicate(self, source: Device, destination: Device) -> bool:
        distance = self._calculate_distance(source, destination)
        path_loss = self._calculate_path_loss(distance)
        
        # Calculate received power
        received_power = source.tx_power - path_loss
        
        # Check if received power is above receiver sensitivity
        return received_power >= destination.rx_sensitivity
        
    # Add traffic between two devices
    def add_traffic(self, source_id: int, destination_id: int, 
                   packet_size: int = 100, interval: float = 1.0) -> None:
        if source_id >= len(self.devices) or destination_id >= len(self.devices):
            raise ValueError("Invalid device ID")
            
        # Add packet to queue
        self.packet_queue.append({
            'source_id': source_id,
            'destination_id': destination_id,
            'packet_size': packet_size,
            'interval': interval,
            'next_transmission': self.simulation_time
        })
        
    # Process a single packet transmission
    def _process_packet(self, packet: Dict) -> None:
        source = self.devices[packet['source_id']]
        destination = self.devices[packet['destination_id']]
        
        # Check if devices can communicate
        if self._can_communicate(source, destination):
            # Update statistics
            source.packets_sent += 1
            source.bytes_sent += packet['packet_size']
            destination.packets_received += 1
            destination.bytes_received += packet['packet_size']
            
            # Update energy consumption
            # Assume 50mW for transmission and 20mW for reception
            transmission_time = packet['packet_size'] * 8 / (250 * 1000)  # 250 kbps
            source.energy -= 0.050 * transmission_time
            destination.energy -= 0.020 * transmission_time
            
        # Schedule next transmission
        packet['next_transmission'] = self.simulation_time + packet['interval']
        
    # Start the network simulation
    def start_simulation(self, duration: float = 100.0) -> None:
        end_time = self.simulation_time + duration
        
        while self.simulation_time < end_time:
            # Process all packets due for transmission
            current_packets = [p for p in self.packet_queue 
                             if p['next_transmission'] <= self.simulation_time]
            
            for packet in current_packets:
                self._process_packet(packet)
                
            # Advance simulation time
            if current_packets:
                self.simulation_time = min(p['next_transmission'] 
                                        for p in current_packets)
            else:
                self.simulation_time = end_time
                
    # Collect data from the network simulation
    def collect_data(self) -> Dict:
        data = {
            'device_positions': [],
            'packet_stats': [],
            'energy_levels': []
        }
        
        # Collect device positions and statistics
        for device in self.devices:
            data['device_positions'].append({
                'device_id': device.id,
                'type': device.type.value,
                'x': device.x,
                'y': device.y,
                'z': device.z
            })
            
            data['packet_stats'].append({
                'device_id': device.id,
                'tx_packets': device.packets_sent,
                'rx_packets': device.packets_received,
                'tx_bytes': device.bytes_sent,
                'rx_bytes': device.bytes_received
            })
            
            data['energy_levels'].append({
                'device_id': device.id,
                'remaining_energy': device.energy
            })
            
        return data
        
    # Stop the network simulation
    def stop_simulation(self) -> None:
        self.packet_queue.clear()
        self.simulation_time = 0.0

if __name__ == "__main__":
    # Example usage
    network = ZigBeeNetwork(num_devices=10)
    network.setup_network()
    
    # Add some traffic
    network.add_traffic(0, 1)  # Device 0 sends to device 1
    network.add_traffic(2, 3)  # Device 2 sends to device 3
    
    # Run simulation
    network.start_simulation(duration=100.0)
    data = network.collect_data()
    network.stop_simulation()
    
    print("Simulation completed. Collected data:")
    print(f"Device positions: {data['device_positions']}")
    print(f"Packet statistics: {data['packet_stats']}")
    print(f"Energy levels: {data['energy_levels']}") 