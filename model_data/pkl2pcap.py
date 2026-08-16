# -*- coding:utf-8 -*-
import os
import pickle
import argparse
import sys
from io import StringIO
import numpy as np
from scapy.all import IP, TCP, UDP, Raw, wrpcap


def load_test_data(data_path):
    """Load test features (and labels, if available).

    Supports both formats:
      - new : a dict {"X": np.ndarray, "Y": np.ndarray}  -> returns (X, Y)
      - old : a list of feature-lists                    -> returns (X, None)
    """
    with open(data_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "X" in obj:          # new test.pkl format
        X = np.asarray(obj["X"])
        Y = np.asarray(obj["Y"]) if obj.get("Y") is not None else None
        print(f'加载数据(新格式dict): {len(X)} 个样本, 每个样本 {X.shape[1]} 个特征')
        return X, Y
    X = np.asarray(obj)                               # old list format
    print(f'加载数据(旧格式list): {len(X)} 个样本, 每个样本 {len(X[0])} 个特征')
    return X, None


def features_to_packet(features, feature_list):
    """
    根据特征构造网络包
    
    特征顺序:
    - Total length: 总长度
    - Protocol: 协议 (6=TCP, 17=UDP, 1=ICMP)
    - IPV4 Flags (DF): IPv4标志DF位 (0或1)
    - Time to live: TTL
    - Src Port: 源端口
    - Dst Port: 目标端口
    - TCP flags (Reset): TCP RST标志 (0或1)
    - TCP flags (Syn): TCP SYN标志 (0或1)
    
    Args:
        features: 特征数组
        feature_list: 特征名称列表
        
    Returns:
        Scapy包对象
    """
    # 创建特征字典以便访问
    feature_dict = {}
    for i, feat_name in enumerate(feature_list):
        feature_dict[feat_name] = features[i]
    
    # 提取特征值
    total_length = int(feature_dict.get('Total length', 60))
    protocol = int(feature_dict.get('Protocol', 6))  # 默认TCP
    df_flag = int(feature_dict.get('IPV4 Flags (DF)', 0))
    ttl = int(feature_dict.get('Time to live', 64))
    src_port = int(feature_dict.get('Src Port', 12345))
    dst_port = int(feature_dict.get('Dst Port', 80))
    tcp_rst = int(feature_dict.get('TCP flags (Reset)', 0))
    tcp_syn = int(feature_dict.get('TCP flags (Syn)', 0))
    
    # 构造IP层
    ip_flags = 0
    if df_flag:
        ip_flags = 0x02  # DF标志位
    
    # 默认源IP和目标IP
    src_ip = "192.168.1.100"
    dst_ip = "192.168.1.200"
    
    # 根据协议类型构造不同的包
    if protocol == 6:  # TCP
        # 构造TCP标志
        tcp_flags = 0
        if tcp_syn:
            tcp_flags |= 0x02  # SYN
        if tcp_rst:
            tcp_flags |= 0x04  # RST
        if not tcp_syn and not tcp_rst:
            tcp_flags = 0x10  # ACK (默认)
        
        # 构造TCP包
        tcp = TCP(sport=src_port, dport=dst_port, flags=tcp_flags)
        packet = IP(src=src_ip, dst=dst_ip, ttl=ttl, flags=ip_flags) / tcp
        
        # 计算需要的数据长度
        ip_header_len = 20
        tcp_header_len = 20
        data_len = max(0, total_length - ip_header_len - tcp_header_len)
        
        if data_len > 0:
            packet = packet / Raw(b'X' * data_len)
    
    elif protocol == 17:  # UDP
        udp = UDP(sport=src_port, dport=dst_port)
        packet = IP(src=src_ip, dst=dst_ip, ttl=ttl, flags=ip_flags, proto=17) / udp
        
        # 计算需要的数据长度
        ip_header_len = 20
        udp_header_len = 8
        data_len = max(0, total_length - ip_header_len - udp_header_len)
        
        if data_len > 0:
            packet = packet / Raw(b'X' * data_len)
    
    else:  # 其他协议，使用IP包
        packet = IP(src=src_ip, dst=dst_ip, ttl=ttl, flags=ip_flags, proto=protocol)
        
        # 计算需要的数据长度
        ip_header_len = 20
        data_len = max(0, total_length - ip_header_len)
        
        if data_len > 0:
            packet = packet / Raw(b'X' * data_len)
    
    return packet


def print_packet_info(packet, features, ground_truth=None, packet_index=0):
    """
    打印包的详细信息
    
    Args:
        packet: Scapy包对象
        features: 原始特征数组
        ground_truth: ground truth标签（可选）
        packet_index: 包索引
    """
    print("\n" + "="*60)
    print(f"第 {packet_index + 1} 个包的信息:")
    print("="*60)
    
    # 打印原始特征
    feature_list = [
        'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
        'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
    ]
    print("\n原始特征:")
    for i, feat_name in enumerate(feature_list):
        print(f"  {feat_name}: {features[i]}")
    
    # 打印ground truth
    if ground_truth is not None:
        print(f"\nGround Truth: {ground_truth}")
    
    # 打印包的详细信息
    print("\n构造的网络包信息:")
    # 捕获show()的输出
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    packet.show()
    sys.stdout = old_stdout
    print(captured_output.getvalue())


def pkl_to_pcap(pkl_path="./univ1/test.pkl",
                 output_path="./univ1/test.pcap",
                 max_packets=None):
    """
    将pkl文件中的特征数据转换为pcap文件

    Args:
        pkl_path: 输入的pkl文件路径 (新 dict {X,Y} 格式, Y 随 pkl 自带)
        output_path: 输出的pcap文件路径
        max_packets: 最大包数量，None表示处理所有包
    """
    # 特征列表（与train_edt.py中保持一致）
    feature_list = [
        'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
        'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
    ]

    # 加载数据 (新 test.pkl 为 dict {X,Y}，Y 随 pkl 自带)
    print(f"正在加载数据: {pkl_path}")
    x_test, y_test = load_test_data(pkl_path)
    if y_test is None:
        raise ValueError(f"{pkl_path} 不含 Y 标签 (旧 list 格式已弃用)；请使用新 dict {X,Y} 格式的 test.pkl")
    print(f"ground truth 已随 {pkl_path} 一并加载: {len(y_test)} 个标签")
    # 转换为numpy数组（如果还不是）
    if not isinstance(x_test, np.ndarray):
        x_test = np.array(x_test)
    
    # 如果指定了max_packets且小于数据集大小，则随机采样
    if max_packets is not None and max_packets < len(x_test):
        # 生成随机索引
        random_indices = np.random.choice(len(x_test), size=max_packets, replace=False)
        x_test = x_test[random_indices]
        y_test = y_test[random_indices]
        print(f"随机抽取 {max_packets} 个包")
    else:
        print(f"处理全部 {len(x_test)} 个包")
    
    # 构造包列表
    print("正在构造网络包...")
    packets = []
    first_packet = None
    first_features = None
    first_ground_truth = None
    
    for i, features in enumerate(x_test):
        try:
            packet = features_to_packet(features, feature_list)
            packets.append(packet)
            
            # 保存第一个包的信息
            if i == 0:
                first_packet = packet
                first_features = features
                first_ground_truth = y_test[i] if i < len(y_test) else None
            
            if (i + 1) % 1000 == 0:
                print(f"已处理 {i + 1}/{len(x_test)} 个包")
        except Exception as e:
            print(f"处理第 {i+1} 个包时出错: {e}")
            continue
    
    # 打印第一个包的信息
    if first_packet is not None:
        print_packet_info(first_packet, first_features, first_ground_truth, 0)
    
    # 写入pcap文件
    print(f"正在写入pcap文件: {output_path}")
    wrpcap(output_path, packets)
    print(f"成功生成pcap文件: {output_path}")
    print(f"共生成 {len(packets)} 个网络包")
    
    # 保存ground truth到文本文件
    gt_output_path = output_path.replace('.pcap', '_ground_truth.txt')
    print(f"\n正在保存ground truth到: {gt_output_path}")
    with open(gt_output_path, 'w') as f:
        f.write("包索引\tGround Truth\n")
        for i, gt in enumerate(y_test[:len(packets)]):
            f.write(f"{i}\t{gt}\n")
    print(f"Ground truth已保存到: {gt_output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将pkl特征数据转换为pcap文件')
    parser.add_argument('--input', '-i',
                       default='./univ1/test.pkl',
                       help='输入的pkl文件路径 (默认: ./univ1/test.pkl, 新 dict {X,Y} 格式)')
    parser.add_argument('--output', '-o',
                       default='./univ1/test.pcap',
                       help='输出的pcap文件路径 (默认: ./univ1/test.pcap)')
    parser.add_argument('--max', '-m',
                       type=int,
                       default=100,
                       help='最大处理包数量 (默认: 100)')
    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        exit(1)

    # 执行转换 (新 dict 格式 test.pkl 自带 Y)
    pkl_to_pcap(args.input, args.output, args.max)