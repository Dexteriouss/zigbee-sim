import numpy as np
import sys
from typing import Dict, List, Optional, Tuple
import time
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Union
import heapq

class PacketType(Enum):
    ACK = 0    # Acknowledgment packet
    REQ = 1    # Request packet
    DATA = 2   # Data packet

class ReturnMsg(Enum):
    SUCCESS = 0
    FAILURE = 1

# Represents a ZigBee device in the network
@dataclass
class Device:
    id: int
    x: float
    y: float
    transmit_power: float
    min_recieve_power: float
    packets_sent: int = 0
    packets_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    packet_queue: List["Packet"] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Device) and self.id == other.id
    
    def __str__(self):
        return f"Device {str(self.id)}"
    
    def __repr__(self):
        return f"{str(self.id)}"

# Coordinator device
@dataclass
class Coordinator(Device):
    neighbors: List[Device] = field(default_factory=list) # Routers
    routing_table: Dict[Device, Dict[Device, Device]] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Coordinator) and self.id == other.id
    
    def __str__(self):
        return f"Device {str(self.id)}"
    
    def __repr__(self):
        return f"{str(self.id)}"
    
# Router device
@dataclass
class Router(Device):
    neighbors: List[Device] = field(default_factory=list) # Routers
    children: List[Device] = field(default_factory=list)# EndDevices

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Router) and self.id == other.id
    
    def __str__(self):
        return f"Device {str(self.id)}"
    
    def __repr__(self):
        return f"{str(self.id)}"

# End Device
@dataclass
class EndDevice(Device):
    parent: Optional[Router] = None

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, EndDevice) and self.id == other.id
    
    def __str__(self):
        return f"Device {str(self.id)}"
    
    def __repr__(self):
        return f"{str(self.id)}"

@dataclass
class Packet:
    packet_type: PacketType
    source: Device
    destination: Device
    packet_size: int        # in bytes
    TTL: int                # in secs
    timestamp: float        # uses time.time_ns(), ns since epoch
    data: Union[str, int, float, Device]


# Initialize a ZigBee network simulation
class ZigBeeNetwork:
    def __init__(self, num_devices: int = 10, area_size: float = 100.0):
        self.num_devices = num_devices
        self.area_size = area_size
        self.devices: List[Device] = []
        self.simulation_time = 0.0
        self.transmit_speed = 250000 # 250 kbps
        self.transmit_power = 0 # dBm
        self.min_recieve_power = -85 # dBm
        
    # Set up the initial network configuration
    def setup_network(self) -> None:
        # Create devices
        for i in range(self.num_devices):
            # Determine position
            x = random.uniform(0, self.area_size)
            y = random.uniform(0, self.area_size)

            # Determine device type
            if i == 0:
                # One coordinator
                device = Coordinator(
                    id=i,
                    # Position
                    x=x,
                    y=y,
                    # Other paramaters
                    transmit_power=self.transmit_power,  # dBm
                    min_recieve_power=self.min_recieve_power,  # dBm
                )
            elif i < self.num_devices // 2: 
                # Half routers
                device = Router(
                    id=i,
                    # Position
                    x=x,
                    y=y,
                    # Other paramaters
                    transmit_power=self.transmit_power,  # dBm
                    min_recieve_power=self.min_recieve_power,  # dBm
                )
            else:
                # Remaining Devices are EndDevices
                device = EndDevice(
                    id=i,
                    # Position
                    x=x,
                    y=y,
                    # Other paramaters
                    transmit_power=self.transmit_power,  # dBm
                    min_recieve_power=self.min_recieve_power,  # dBm
                )
            
            # Add to device list
            self.devices.append(device)
            
        # Build neighbor lists based on distance
        self.build_neighbor_lists()
        
    # Build neighbor lists for each device based on distance
    def build_neighbor_lists(self) -> None:
        for i, device1 in enumerate(self.devices):
            max_power = -1000000
            for j, device2 in enumerate(self.devices):
                # Build communication network between routers and the coordinator
                if (i != j and 
                    isinstance(device1, (Router, Coordinator)) and 
                    isinstance(device2, (Router, Coordinator)) and 
                    self.can_communicate(device1, device2)):
                    
                    # Add to neighbors list
                    device1.neighbors.append(device2)

                # For every EndDevice, set the router with the most power as the parent to route traffic through
                elif (i != j and
                    isinstance(device1, (EndDevice)) and 
                    isinstance(device2, (Router)) and 
                    self.can_communicate(device1, device2)):

                    if (max_power < self.recieved_power(device1, device2)):
                        max_power = self.recieved_power(device1, device2)
                        if (device1.parent):
                            device1.parent.children.remove(device1)
                        device1.parent = device2
                        device2.children.append(device1)
                        
    # Calculate distance between two devices
    def calculate_distance(self, device1: Device, device2: Device) -> float:
        return np.sqrt(
            (device1.x - device2.x)**2 +
            (device1.y - device2.y)**2
        )

    # Calculate Free-Space Path Loss (SP11)
    def calculate_path_loss(self, distance: float) -> float:
        path_loss_exponent = 2.0 # Path loss exponent (typically 2-4)
        d0 = 1.0 # Reference distance (1 meter)
        PL0 = 40.0 # Path loss at reference distance (dB)
        
        return PL0 + 10 * path_loss_exponent * np.log10(distance / d0)
        
    # Check if received power is above the reciever's minimum power threshold
    def can_communicate(self, source: Device, destination: Device) -> bool:
        return self.recieved_power(source, destination) >= destination.min_recieve_power

    # Determine if two devices can communicate based on distance and power
    def recieved_power(self, source: Device, destination: Device) -> float:
        # Calculate path loss
        distance = self.calculate_distance(source, destination)
        path_loss = self.calculate_path_loss(distance)
        
        # Calculate received power
        return source.transmit_power - path_loss

    # Build a packet
    def build_packet(self, type: PacketType, source: Device, destination: Device, packet_size: int, TTL: int, data: Union[str, int, float, Device]) -> Packet:
        if source not in self.devices or destination not in self.devices:
            raise ValueError("Invalid device ID")
        
        timestamp = time.time_ns()
        return Packet(type, source, destination, packet_size, TTL, timestamp, data)
    
    # Queue a packet for a device
    def queue_packet(self, device: Device, packet: Packet):
        device.packet_queue.append(packet)

    # Process a single packet transmission
    def process_packet(self, device: Device, packet: Packet) -> None:
        print(f"Device {device.id}: Processing a packet: {packet.source} to {packet.destination}")
        # Check if packet is still valid
        if (time.time_ns() - (packet.TTL * 1000000000) >= packet.timestamp and device != packet.source):
            print(f"Packet died at {device}: {time.time_ns() - (packet.TTL * 1000000000)}, {packet.timestamp}")
            device.packet_queue.remove(packet)
            return ReturnMsg.FAILURE
        # Check if the device is connected to the network
        elif ((isinstance(device, EndDevice) and not device.parent)):
            print("Device not connected to network.")
            device.packet_queue.remove(packet)
            return ReturnMsg.FAILURE
        elif (isinstance(device, (Coordinator, Router)) and not device.neighbors):
            print("Device not connected to network.")
            device.packet_queue.remove(packet)
            return ReturnMsg.FAILURE
        elif (isinstance(packet.destination, (Coordinator, Router))and not packet.destination.neighbors):
            print("Destination not connected to network.")
            device.packet_queue.remove(packet)
            return ReturnMsg.FAILURE
        elif (isinstance(packet.destination, (EndDevice))and not packet.destination.parent):
            print("Destination not connected to network.")
            device.packet_queue.remove(packet)
            return ReturnMsg.FAILURE
        
        # Reaches destination
        if (packet.destination == device):
            # Packet logic for sending route information - future
            print(f"Packet recieved at {device}.\n")
            device.packet_queue.remove(packet)
            return ReturnMsg.SUCCESS

        # First transmission
        elif (packet.source == device):
            self.transmit_packet(device, device.parent, packet)
            return ReturnMsg.SUCCESS
        
        # Packet reaches parent router of destination
        elif (isinstance(device, Router) and (packet.destination in device.children or packet.destination in device.neighbors)):
            self.transmit_packet(device, packet.destination, packet)
            return ReturnMsg.SUCCESS
        
        # Intermediate transmission
        else:
            # Future implementation - have to search for coordinator
            device_to_send = self.get_routing(self.devices[0], device, packet.destination)
            self.transmit_packet(device, device_to_send, packet)
            return ReturnMsg.SUCCESS 
    
    # Makes a request to the Coordinator for routing
    def get_routing(self, coordinator: Coordinator, requesting_device: Device, device_to_search: Device) -> Device:
        temppacket = self.build_packet(PacketType.REQ, requesting_device, coordinator, 100, 5, device_to_search)
        packet = self.build_packet(PacketType.REQ, requesting_device, coordinator, sys.getsizeof(temppacket), 5, device_to_search)
        self.queue_packet(requesting_device, packet)
        self.transmit_packet(packet.source, packet.destination, packet)

        # Gets route from Coordinator automatically - develop in future
        try:
            if (isinstance(device_to_search, Router)):
                print(f"Searching for {device_to_search}:", coordinator.routing_table[requesting_device][device_to_search])
                return coordinator.routing_table[requesting_device][device_to_search]
            elif (isinstance(device_to_search, EndDevice)):
                print(f"Searching for {device_to_search.parent}:", coordinator.routing_table[requesting_device][device_to_search.parent])
                return coordinator.routing_table[requesting_device][device_to_search.parent]
            return coordinator.routing_table[requesting_device][device_to_search]
        except AttributeError:
            print("Network not connected to coordinator.")

    # Transmits a packet from a source to destination
    def transmit_packet(self, source: Device, destination: Device, packet: Packet):
        source.packet_queue.remove(packet)
        # Set timestamp for initial transmission
        if (source == packet.source):
            packet.timestamp = time.time_ns()
        destination.packet_queue.append(packet)
        print(f"Transmitting a packet: {source.id} to {destination.id}.")

        time.sleep((packet.packet_size * 8)/(self.transmit_speed))

        print(f"Transmission complete.")

    def build_routing_table(self, coordinator: Coordinator):
        # Initialize routing table
        coordinator.routing_table = {}

        for beginning_device in self.devices:
            if not (isinstance(beginning_device, EndDevice)):
                for target in self.devices:
                    if target == beginning_device:
                        continue

                    # Dijkstra's algorithm from device to target
                    distances = {device: float('inf') for device in self.devices}
                    previous = {device: None for device in self.devices}
                    distances[beginning_device] = 0

                    heap = [(0, beginning_device)]

                    while heap:
                        current_dist, current_device = heapq.heappop(heap)

                        if current_dist > distances[current_device]:
                            continue

                        for neighbor in current_device.neighbors:
                            if neighbor not in distances:
                                continue  # Skip unknown nodes

                            alt = current_dist + self.calculate_distance(current_device, neighbor)
                            if alt < distances[neighbor]:
                                distances[neighbor] = alt
                                previous[neighbor] = current_device
                                heapq.heappush(heap, (alt, neighbor))

                    # Backtrack from target to find next hop
                    path = []
                    step = target 
                    # Reconstruct path only if there's a path
                    while step and step != beginning_device:
                        if previous[step] is None:
                            path = []  # Clear path to indicate no route
                            break
                        path.append(step)
                        step = previous[step]

                    path.reverse()

                    if path:
                        next_hop = path[0]
                        if beginning_device not in coordinator.routing_table:
                            coordinator.routing_table[beginning_device] = {}
                        coordinator.routing_table[beginning_device][target] = next_hop
                    
        return coordinator.routing_table


