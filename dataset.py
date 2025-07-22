import os
from typing import Tuple, List
import random
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as Data
from sklearn.decomposition import PCA
from einops import rearrange
from sklearn.preprocessing import RobustScaler

def setup_seed(seed: int) -> None:
    """Set up random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_pca(X: np.ndarray, num_components: int = 75) -> np.ndarray:
    """Apply PCA to reduce the number of spectral bands."""
    reshaped_X = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_X = pca.fit_transform(reshaped_X)
    return np.reshape(new_X, (X.shape[0], X.shape[1], num_components))


def load_houston_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Houston dataset."""
    HSI_data = sio.loadmat(os.path.join(data_path, 'Houston2013/HSI.mat'))['HSI']
    LiDAR_data = sio.loadmat(os.path.join(data_path, 'Houston2013/LiDAR.mat'))['LiDAR']
    LiDAR_data = np.expand_dims(LiDAR_data, axis=-1)
    Train_data = sio.loadmat(os.path.join(data_path, 'Houston2013/TRLabel.mat'))['TRLabel']
    Test_data = sio.loadmat(os.path.join(data_path, 'Houston2013/TSLabel.mat'))['TSLabel']
    GT = sio.loadmat(os.path.join(data_path, 'Houston2013/gt.mat'))['gt']
    return HSI_data, LiDAR_data, Train_data, Test_data, GT


def load_berlin_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Berlin dataset."""
    HSI_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/data_HS_LR.mat'))['data_HS_LR']
    LiDAR_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/data_SAR_HR.mat'))['data_SAR_HR']
    Train_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/TrainImage.mat'))['TrainImage']
    Test_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/TestImage.mat'))['TestImage']
    GT = Train_data + Test_data
    return HSI_data, LiDAR_data, Train_data, Test_data, GT


def load_trento_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Trento dataset."""
    HSI_data = sio.loadmat(os.path.join(data_path, 'Trento/HSI.mat'))['HSI']
    LiDAR_data = sio.loadmat(os.path.join(data_path, 'Trento/LiDAR.mat'))['LiDAR']
    LiDAR_data = np.expand_dims(LiDAR_data, axis=-1)
    Train_data = sio.loadmat(os.path.join(data_path, 'Trento/TRLabel.mat'))['TRLabel']
    Test_data = sio.loadmat(os.path.join(data_path, 'Trento/TSLabel.mat'))['TSLabel']
    GT = sio.loadmat(os.path.join(data_path, 'Trento/gt.mat'))['gt']
    return HSI_data, LiDAR_data, Train_data, Test_data, GT


def load_muufl_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MUUFL dataset."""
    HSI_data = sio.loadmat(os.path.join(data_path, 'MUUFL/HSI.mat'))['HSI']
    LiDAR_data = sio.loadmat(os.path.join(data_path, 'MUUFL/LiDAR.mat'))['LiDAR']
    Train_data = sio.loadmat(os.path.join(data_path, 'MUUFL/mask_train_150.mat'))['mask_train']
    Test_data = sio.loadmat(os.path.join(data_path, 'MUUFL/mask_test_150.mat'))['mask_test']
    GT = sio.loadmat(os.path.join(data_path, 'MUUFL/gt.mat'))['gt']
    GT[GT == -1] = 0  # Replace -1 values with 0
    return HSI_data, LiDAR_data, Train_data, Test_data, GT



def load_augsburg_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Augsburg dataset."""
    HSI_data = sio.loadmat(os.path.join(data_path, 'HS-SAR-DSM Augsburg/data_HS_LR.mat'))['data_HS_LR']
    LiDAR_data = sio.loadmat(os.path.join(data_path, 'HS-SAR-DSM Augsburg/data_DSM.mat'))['data_DSM']
    LiDAR_data = np.expand_dims(LiDAR_data, axis=-1)
    Train_data = sio.loadmat(os.path.join(data_path, 'HS-SAR-DSM Augsburg/TrainImage.mat'))['TrainImage']
    Test_data = sio.loadmat(os.path.join(data_path, 'HS-SAR-DSM Augsburg/TestImage.mat'))['TestImage']
    GT = Train_data + Test_data
    return HSI_data, LiDAR_data, Train_data, Test_data, GT

def load_Houston2018_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Houston2018 dataset.
    
    Args:
        data_path: 数据集根目录路径
        
    Returns:
        HSI_data: 高光谱图像数据
        LiDAR_data: LiDAR数据
        Train_data: 训练集标签
        Test_data: 测试集标签
        GT: 地面真值数据
    """
    # 加载高光谱数据
    HSI_data = sio.loadmat(os.path.join(data_path, 'Houston2018/houston_hsi.mat'))['houston_hsi']
    
    # 加载LiDAR数据
    LiDAR_data = sio.loadmat(os.path.join(data_path, 'Houston2018/houston_lidar.mat'))['houston_lidar']
    LiDAR_data = np.expand_dims(LiDAR_data, axis=-1)  # 添加通道维度
    
    # 加载地面真值
    GT = sio.loadmat(os.path.join(data_path, 'Houston2018/houston_gt.mat'))['houston_gt']
    
    # 加载训练和测试索引
    index_dict = sio.loadmat(os.path.join(data_path, 'Houston2018/houston_index.mat'))
    train_indices = index_dict['houston_train'] - 1  # MATLAB索引从1开始，转换为Python的0基索引
    test_indices = index_dict['houston_test'] - 1
    
    # 创建训练集和测试集掩码
    Train_data = np.zeros_like(GT)
    Test_data = np.zeros_like(GT)
    
    # 使用索引填充训练集
    for idx in train_indices:
        Train_data[idx[0], idx[1]] = GT[idx[0], idx[1]]
    
    # 使用索引填充测试集
    for idx in test_indices:
        Test_data[idx[0], idx[1]] = GT[idx[0], idx[1]]
    
    return HSI_data, LiDAR_data, Train_data, Test_data, GT



def load_data(dataset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset based on the dataset name."""
    data_path = r'/home/ubuntu/dataset_RS/Multisource/data'
    dataset_loaders = {
        'Houston': load_houston_data,
        'Berlin': load_berlin_data,
        'Trento': load_trento_data,
        'MUUFL': load_muufl_data,
        'Augsburg': load_augsburg_data,
        'Houston2018': load_Houston2018_data
    }

    if dataset not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Available options are: {', '.join(dataset_loaders.keys())}")

    return dataset_loaders[dataset](data_path)


def generate_patches(data1: np.ndarray, data2: np.ndarray, patchsize: int, pad_width: int, label: np.ndarray) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate patches for training from two data sources."""
    m1, n1, l1 = data1.shape
    m2, n2, l2 = data2.shape

    x1_pad = np.pad(data1, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='symmetric')
    x2_pad = np.pad(data2, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='symmetric')

    ind1, ind2 = np.where(label > 0)
    train_num = len(ind1)

    train_patch1 = np.empty((train_num, l1, patchsize, patchsize), dtype='float32')
    train_patch2 = np.empty((train_num, l2, patchsize, patchsize), dtype='float32')
    train_label = np.empty(train_num)

    for i in range(train_num):
        r, c = ind1[i] + pad_width, ind2[i] + pad_width
        train_patch1[i] = np.transpose(x1_pad[r - pad_width:r + pad_width, c - pad_width:c + pad_width, :], (2, 0, 1))
        train_patch2[i] = np.transpose(x2_pad[r - pad_width:r + pad_width, c - pad_width:c + pad_width, :], (2, 0, 1))
        train_label[i] = label[ind1[i], ind2[i]]

    return torch.from_numpy(train_patch1), torch.from_numpy(train_patch2), torch.from_numpy(train_label) - 1


def shape_new_X(x: np.ndarray) -> np.ndarray:
    """Flip the odd rows in the input matrix for spectral augmentation."""
    x[:, 1::2, :] = np.flip(x[:, 1::2, :], axis=-1)
    return x


def pixel_select(GT: np.ndarray, Y: np.ndarray, train_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    按照比例为每个类别选择训练样本
    
    Args:
        Y: 输入的标签数组
        train_ratio: 训练集占总样本的比例，默认为0.2（20%）
    
    Returns:
        train_pixels: 训练集掩码
        test_pixels: 测试集掩码
    """
    test_pixels = Y.copy()
    classes = np.unique(Y).shape[0] - 1  # 减1是因为包含了背景类（0）
    
    for i in range(classes):
        coords = np.where(Y == (i + 1))  # 获取当前类别的所有像素坐标
        num_samples = len(coords[0])
        
        # 计算当前类别需要的训练样本数量
        train_num = 20 #max(1, int(num_samples * train_ratio))  至少选择1个样本
        
        if train_num >= num_samples:
            print(f"警告：第{i+1}类的样本数量（{num_samples}）过少，将使用所有样本作为训练集")
            train_num = num_samples
            
        # 随机选择训练样本的索引
        indices = random.sample(range(num_samples), train_num)
        # 将选中的训练样本在test_pixels中置0
        test_pixels[coords[0][indices], coords[1][indices]] = 0

    # 生成训练集掩码
    train_pixels = Y - test_pixels
    return train_pixels, test_pixels


def get_image_cubes(X: np.ndarray, Y: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract image cubes and corresponding labels."""
    pad_width = window_size // 2
    X_padded = np.pad(X, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='symmetric')

    indices = np.where(Y > 0)
    num_samples = len(indices[0])

    image_cubes = np.zeros((num_samples, X.shape[2], window_size, window_size), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        r, c = indices[0][i] + pad_width, indices[1][i] + pad_width
        image_cubes[i] = np.transpose(X_padded[r - pad_width:r + pad_width + 1, c - pad_width:c + pad_width + 1, :],
                                      (2, 0, 1))
        labels[i] = Y[indices[0][i], indices[1][i]] - 1

    return image_cubes, labels


def generate_data_loaders(X_hsi: np.ndarray, X_lidar: np.ndarray, train_pixels: np.ndarray, test_pixels: np.ndarray, GT: np.ndarray,
                          batch_size: int, window_size: int) -> Tuple[int, int, int, Data.DataLoader, Data.DataLoader]:
    """Generate training and testing data loaders."""
    train_pixels, test_pixels = pixel_select(GT, train_pixels, train_ratio=0.1)
    x_train_hsi, y_train_hsi = get_image_cubes(X_hsi, train_pixels, window_size)
    x_test_hsi, y_test_hsi = get_image_cubes(X_hsi, test_pixels, window_size)

    x_train_lidar, _ = get_image_cubes(X_lidar, train_pixels, window_size)
    x_test_lidar, _ = get_image_cubes(X_lidar, test_pixels, window_size)

    train_size, test_size = x_train_hsi.shape[0], x_test_hsi.shape[0]
    total_size = train_size + test_size

    print(f"Train Size: {train_size}, Test Size: {test_size}, Total: {total_size}")

    train_dataset = Data.TensorDataset(torch.from_numpy(x_train_hsi).float(),
                                       torch.from_numpy(x_train_lidar).float(),
                                       torch.from_numpy(y_train_hsi).long())

    test_dataset = Data.TensorDataset(torch.from_numpy(x_test_hsi).float(),
                                      torch.from_numpy(x_test_lidar).float(),
                                      torch.from_numpy(y_test_hsi).long())

    # 根据GPU数量和可用性设置num_workers
    num_workers = min(4 * torch.cuda.device_count(), 8) if torch.cuda.is_available() else 0
    
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = Data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_size, test_size, total_size, train_loader, test_loader

def robust_normalize(X: np.ndarray) -> np.ndarray:
    """使用Robust标准化处理HSI数据"""
    from sklearn.preprocessing import RobustScaler
    original_shape = X.shape
    # 重塑为2D数组 (pixels, bands)
    X_reshaped = X.reshape(-1, X.shape[-1])
    # 应用Robust标准化
    scaler = RobustScaler()
    X_normalized = scaler.fit_transform(X_reshaped)
    # 恢复原始形状
    return X_normalized.reshape(original_shape)

def enhanced_minmax_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """改进的Min-Max归一化，适用于LiDAR数据"""
    X_min = np.min(X, axis=(0, 1))
    X_max = np.max(X, axis=(0, 1))
    # 添加eps避免除零
    return (X - X_min) / (X_max - X_min + eps)

def normalize(X: np.ndarray, method: int = 1, modality: str = 'HSI') -> np.ndarray:
    """增强的归一化函数
    
    Args:
        X: 输入数据
        method: 归一化方法 (1: robust, 2: enhanced_minmax, 3: z-score)
        modality: 数据模态类型 ('HSI' or 'LiDAR')
    """
    if modality == 'HSI':
        if method == 1:
            return robust_normalize(X)
        elif method == 2:
            return enhanced_minmax_normalize(X)
        elif method == 3:
            return (X - np.mean(X, axis=(0, 1))) / (np.std(X, axis=(0, 1)) + 1e-8)
    else:  # LiDAR
        return enhanced_minmax_normalize(X)
    

