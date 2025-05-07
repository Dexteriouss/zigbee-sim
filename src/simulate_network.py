# run_network.py
import time
import random
import network as net

def main():
    # Initialize network with 10 devices (1 Coordinator, 4 Routers, 5 EndDevices)
    network = net.ZigBeeNetwork(num_devices=10, area_size=200)
    network.setup_network()
    
    # Get references to important devices
    coordinator = network.devices[0]
    routers = [d for d in network.devices if isinstance(d, net.Router)]
    end_devices = [d for d in network.devices if isinstance(d, net.EndDevice)]
    
    print("Starting network simulation...")
    
    for device in network.devices:
        print(device)
    for device in network.devices:
        if not isinstance(device, net.EndDevice):
            print(device.neighbors)
    
    print(network.build_routing_table(network.devices[0]))
    var = True
    try:
        while True:
            # Generate random traffic every 2 seconds
            time.sleep(1)
            while (var):
              # Randomly select an EndDevice to send data
              source = random.choice(end_devices)
              packet = network.build_packet(
                  type=net.PacketType.DATA,
                  source=source,
                  destination=coordinator,
                  packet_size=100,        # bytes
                  TTL=5,                  # seconds
                  data="Sensor Data"      # Example payload
              )
              var = False
              print(f"\nDevice {source.id} sending packet to Coordinator")
              network.queue_packet(source, packet)
            
            # Process packets in all device queues
            for device in network.devices:
                print(f"Device: {device.id}", f"Packets?:", bool(device.packet_queue))
                while device.packet_queue:
                    packet = device.packet_queue[0]
                    result = network.process_packet(device, packet)
                    
                    if result == net.ReturnMsg.SUCCESS:
                        if isinstance(device, net.Coordinator):
                            print(f"Coordinator received: {packet.data}")
                    else:
                        print(f"Packet from {packet.source.id} failed!")
            

    except KeyboardInterrupt:
        print("\nSimulation stopped")

if __name__ == "__main__":
    main()