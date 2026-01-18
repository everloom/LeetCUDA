import torch

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"显卡: {props.name}")
        print(f"架构: {props.major}.{props.minor}")
        print(f"SM 数量: {props.multi_processor_count}")
else:
    print("未检测到 CUDA 设备")