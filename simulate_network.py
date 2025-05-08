# run_network.py
import time
import random
import network as net
import matplotlib.pyplot as plt

PACKET_SIZE = 10000
RESULTSFILE = 'results.txt'

def initialize_network(num_devices: int = 10, area_size: float = 200.0):
    # Initialize network with 10 devices (1 Coordinator, 4 Routers, 5 EndDevices)
    network = net.ZigBeeNetwork(num_devices, area_size)
    network.setup_network()
    print("Intializing network...")
    
    for device in network.devices:
        print(f"{device}, Type: {type(device)}")
    for device in network.devices:
        if not isinstance(device, net.EndDevice):
            print(f"{device}, Neighbors: {device.neighbors}")
        elif isinstance(device, net.EndDevice):
            print(f"{device}, Parent: {device.parent}")
    print(f"Routing Table: {network.build_routing_table(network.devices[0])}")
    return network

def generate_reg_traffic(network: net.ZigBeeNetwork, num_packets: int, ttl: int):
    end_devices = [d for d in network.devices if isinstance(d, net.EndDevice)]

    # Generate packets
    for i in range(num_packets):
        source = random.choice(end_devices)
        destination = random.choice(end_devices)

        while (source == destination):
            destination = random.choice(end_devices)
        packet = network.build_packet(
            type=net.PacketType.DATA,
            source=source,
            destination=destination,
            packet_size=PACKET_SIZE,     # bytes
            TTL=ttl,                     # seconds
            data="Example data"          # Example payload
        )
        network.queue_packet(source, packet)
    
def generate_overload_traffic(network: net.ZigBeeNetwork, num_packets: int, ttl: int):
    end_devices = [d for d in network.devices if isinstance(d, net.EndDevice)]
    attacker = end_devices[-1]
    # Generate real packets
    for i in range(num_packets):
        source = random.choice(end_devices[0:-1])
        destination = random.choice(end_devices[0:-1])

        while (source == destination):
            destination = random.choice(end_devices)
        packet = network.build_packet(
            type=net.PacketType.DATA,
            source=source,
            destination=destination,
            packet_size=PACKET_SIZE,          # bytes
            TTL=ttl,                     # seconds
            data="Real Data"            # Example payload
        )
        network.queue_packet(source, packet)

    # Generate fake packets
    for i in range(num_packets*2):
        destination = random.choice(end_devices[0:-1])
        packet = network.build_packet(
            type=net.PacketType.DATA,
            source=attacker,
            destination=destination,
            packet_size=PACKET_SIZE,    # bytes
            TTL=ttl,                    # seconds
            data="Fake Data"            # Example payload
        )
        network.queue_packet(attacker, packet)

def simulate(network: net.ZigBeeNetwork):
    # Search through all packets queued
    failed_packets = 0
    empty_devices = 0
    try:
        while(empty_devices < len(network.devices)):
          empty_devices = 0
          time.sleep(0.1)
          for device in network.devices:
                print(f"Device: {device.id}", f"Packets:", bool(device.packet_queue), empty_devices, len(network.devices))
                if (device.packet_queue):
                    while device.packet_queue:
                        packet = device.packet_queue[0]
                        result = network.process_packet(device, packet)
                        
                        if result == net.ReturnMsg.SUCCESS:
                            if isinstance(device, net.EndDevice):
                                print(f"{device} processed: {packet.data}")
                        else:
                            failed_packets += 1
                            print(f"Packet from {packet.source} failed!\n")
                            print(failed_packets)
                else:
                    empty_devices += 1
        with open(RESULTSFILE, "a") as f:
            f.write(f"{failed_packets}")
        print(f"Simulation Ended.")
    except KeyboardInterrupt:
        print(f"Simulation Aborted.")

def plot_devices(devices: list):
    plt.figure(figsize=(10, 8))

    # Separate device types
    coordinators = [d for d in devices if isinstance(d, net.Coordinator)]
    routers = [d for d in devices if isinstance(d, net.Router) and not isinstance(d, net.Coordinator)]
    end_devices = [d for d in devices if isinstance(d, net.EndDevice)]

    # Plot Coordinators
    for c in coordinators:
        plt.scatter(c.x, c.y, color='red', label='Coordinator' if 'Coordinator' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(c.x, c.y, f"{c.id}", fontsize=9, ha='right', va='bottom')

        for neighbor in c.neighbors:
            plt.plot([c.x, neighbor.x], [c.y, neighbor.y], color='red', linestyle='--')

    # Plot Routers
    for r in routers:
        plt.scatter(r.x, r.y, color='blue', label='Router' if 'Router' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(r.x, r.y, f"{r.id}", fontsize=9, ha='right', va='bottom')

        for neighbor in r.neighbors:
            if (isinstance(neighbor, net.Router)):
              plt.plot([r.x, neighbor.x], [r.y, neighbor.y], color='blue', linestyle='--')

        for child in r.children:
            plt.plot([r.x, child.x], [r.y, child.y], color='green', linestyle=':')

    # Plot EndDevices
    for e in end_devices:
        plt.scatter(e.x, e.y, color='green', label='EndDevice' if 'EndDevice' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(e.x, e.y, f"{e.id}", fontsize=9, ha='right', va='bottom')

    # Finalize
    plt.title("Device Graph")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig("network_topology.png")
    

if __name__ == "__main__":
    with open(RESULTSFILE, "w") as f:
        f.write("FINAL FAILURE METRICS:\n")
        f.write("\nRegular Fails:")

    network = initialize_network(10, 250)
    generate_reg_traffic(network, num_packets=10, ttl=6)
    simulate(network)

    with open(RESULTSFILE, "a") as f:
        f.write("\n\nAttack Fails:")

    print("Starting attack...")
    time.sleep(3)

    generate_overload_traffic(network, num_packets=20, ttl=6)
    simulate(network)
    plot_devices(network.devices)

