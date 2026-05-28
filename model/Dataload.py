import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def load_data(file_path, device='cpu'):
    """
    加载预处理好的数据并转换为PyTorch Tensor

    参数:
        file_path: 数据文件路径
        device: 设备 ('cpu' 或 'cuda')

    返回:
        tensor_data_dict: 包含所有Tensor数据和配置的字典
    """
    # 1. 加载数据
    data_dict = torch.load(file_path, weights_only=False)

    # 2. 创建Tensor字典
    tensor_dict = {}

    # 转换特征数据
    tensor_dict['x_train'] = torch.FloatTensor(data_dict['x_train']).to(device)
    tensor_dict['x_val'] = torch.FloatTensor(data_dict['x_val']).to(device)
    tensor_dict['x_test'] = torch.FloatTensor(data_dict['x_test']).to(device)


    # 转换标签数据
    tensor_dict['y_train'] = torch.FloatTensor(data_dict['y_train']).to(device)
    tensor_dict['y_val'] = torch.FloatTensor(data_dict['y_val']).to(device)
    tensor_dict['y_test'] = torch.FloatTensor(data_dict['y_test']).to(device)

    #转换ID信息
    tensor_dict['val_ids'] = torch.FloatTensor(data_dict['val_ids']).to(device)
    # 保留配置信息
    tensor_dict['config'] = data_dict['config']

    # 验证转换是否成功
    print(f"\n数据已转换为Tensor格式:")
    print(f"x_train: {tensor_dict['x_train'].dtype}, device: {tensor_dict['x_train'].device}")
    print(f"y_train: {tensor_dict['y_train'].dtype}, device: {tensor_dict['y_train'].device}")

    return tensor_dict


if __name__ == "__main__":
    # 设定数据路径
    path = './train_data/FD001_train_data_a_0.1_seq_30_thresh_130.pt'
    # 加载和转换数据
    tensor_data_dict = load_data(path, device='cpu')
    x_train = tensor_data_dict['x_train']
    y_train = tensor_data_dict['y_train']
    x_val = tensor_data_dict['x_val']
    y_val = tensor_data_dict['y_val']
    x_test = tensor_data_dict['x_test']
    y_test = tensor_data_dict['y_test']

    # 打印数据形状
    print(f"\n数据形状:")
    print(f"x_train: {x_train.shape} (batch, channels, timesteps)")
    print(f"y_train: {y_train.shape} (batch, 1)")
    print(f"x_val: {x_val.shape} (batch, channels, timesteps)")
    print(f"y_val: {y_val.shape} (batch, 1)")
    print(f"x_test: {x_test.shape} (batch, channels, timesteps)")
    print(f"y_test: {y_test.shape} (batch, 1)")

    # 可选：检查配置信息
    print(f"\n配置信息: {tensor_data_dict['config']}")
