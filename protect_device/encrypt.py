import subprocess
#import pyshark
import numpy as np
from scapy.all import sendp, Ether, IP, TCP, get_if_hwaddr, get_if_list

def get_connected_wifi_mac(interface='wlan0'):
    # 获取当前连接的WiFi的MAC地址
    result = subprocess.run(['iwgetid', interface, '--raw'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Failed to get connected WiFi MAC address")
        return None
    ssid = result.stdout.strip()
    if not ssid:
        print("Not connected to any WiFi network")
        return None

    # 获取路由器MAC地址
    result = subprocess.run(['iwlist', interface, 'scan'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if ssid in line:
            for mac_line in result.stdout.split('\n'):
                if "Address:" in mac_line:
                    return mac_line.split("Address: ")[1].strip()
    return None

def change_mac_address(interface, new_mac):
    # 更改本地MAC地址
    subprocess.run(['sudo', 'ifconfig', interface, 'down'])
    subprocess.run(['sudo', 'ifconfig', interface, 'hw', 'ether', new_mac])
    subprocess.run(['sudo', 'ifconfig', interface, 'up'])

def capture_csi(interface='wlan0'):
    capture = pyshark.LiveCapture(interface=interface)
    for packet in capture.sniff_continuously(packet_count=10):
        if 'WLAN' in packet:
            # 假设CSI数据在此处提取
            csi_data = packet.wlan.tfs
            return csi_data.encode()

def generate_encryption_matrix(num_tx, num_rx):
    # 生成一个随机的加密矩阵Ψ
    return np.random.randn(num_tx, num_rx) + 1j * np.random.randn(num_tx, num_rx)

def encrypt_csi(csi_data, encryption_matrix):
    # 假设csi_data是一个numpy数组
    csi_matrix = np.array(csi_data)
    encrypted_csi = np.dot(encryption_matrix, csi_matrix)
    return encrypted_csi

def send_encrypted_data(data, host='192.168.0.1', port=8080):
    pkt = Ether()/IP(dst=host)/TCP(dport=port)/data.tobytes()
    sendp(pkt)

def start_fake_ap():
    subprocess.run(['sudo', 'hostapd', 'hostapd.conf'])

def main():
    # 设置网络接口名称
    interface = 'wlan0'

    # 获取当前连接的WiFi的MAC地址
    wifi_mac = get_connected_wifi_mac(interface)
    if not wifi_mac:
        print("Failed to get connected WiFi MAC address")
        return
    print(f"Connected WiFi MAC address: {wifi_mac}")

    # 更改本地MAC地址
    original_mac = get_if_hwaddr(interface)
    print(f"Original MAC address: {original_mac}")
    change_mac_address(interface, wifi_mac)
    new_mac = get_if_hwaddr(interface)
    print(f"New MAC address: {new_mac}")

    # 启动伪AP
    print("Starting Fake AP...")
    start_fake_ap()

    # 抓取CSI数据
    csi_data = capture_csi(interface)
    if not csi_data:
        print("No CSI data captured")
        return

    print(f"Captured CSI: {csi_data}")

    # 生成加密矩阵Ψ
    encryption_matrix = generate_encryption_matrix(3, 3)

    # 加密CSI数据
    encrypted_csi = encrypt_csi(csi_data, encryption_matrix)
    print(f"Encrypted CSI:\n{encrypted_csi}")

    # 发送加密CSI数据
    send_encrypted_data(encrypted_csi)

if __name__ == "__main__":
    main()
