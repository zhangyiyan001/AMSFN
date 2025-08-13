# -*- coding: utf-8 -*-
import os  
import time  
import yaml  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import numpy as np 
import argparse  
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score 
from torch.optim.lr_scheduler import CosineAnnealingLR 
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from dataset import load_data, generate_data_loaders, normalize, setup_seed  
from module import AA_andEachClassAccuracy  
from vmamba import MultimodalClassifier  
 

def get_dataset_info(dataset_name):
   
    dataset_info = {
        'Houston': {  
            'num_classes': 15,  
            'HSI_Channel': 144,  
            'LiDAR_Channel': 1,  
            'target_names': ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Tree', 'Soil', 'Water',
                           'Residential', 'Commercial', 'Road', 'Highway', 'Railway', 'Parking lot 1',
                           'Parking lot 2', 'Tennis court', 'Running track'] 
        },
        'MUUFL': {  
            'num_classes': 11,
            'HSI_Channel': 64,
            'LiDAR_Channel': 2,
            'target_names': ['Trees', 'Mostly grass', 'Mixed ground surface', 'Dirt and sand', 'Road', 'Water',
                           'Building shadow', 'Building', 'Sidewalk', 'Yellow curb', 'Cloth panels']
        },
        'Trento': {  
            'num_classes': 6,
            'HSI_Channel': 63,
            'LiDAR_Channel': 1,
            'target_names': ['Apple trees', 'Buildings', 'Ground', 'Wood', 'Vineyard', 'Roads']
        },
       
        'Augsburg': {  
            'num_classes': 7,
            'HSI_Channel': 180,
            'LiDAR_Channel': 1,
            'target_names': ['Forest', 'Building', 'Low Vegetation', 'Car', 'Clutter', 'Soil', 'Water']
        },
        'Houston2018': {
            'num_classes': 20,
            'HSI_Channel': 50,
            'LiDAR_Channel': 1,
            'target_names': ['Healthy grass', 'Stressed grass', 'Artificial turf', 'Evergreen trees',
                           'Deciduous trees', 'Bare earth', 'Water', 'Residential buildings', 'Non-residential buildings',
                           'Roads', 'Sidewalks', 'Crosswalks', 'Major thoroughfares', 'Highways', 'Railways',
                           'Paved parking lots', 'Unpaved parking lots', 'Cars', 'Trains', 'Stadium seats']
        }
    }
    return dataset_info.get(dataset_name)  

def prepare_data(args):
    HSI_data, LiDAR_data, Train_data, Test_data, GT = load_data(args.dataset)
   
    HSI_data = normalize(HSI_data, method=1, modality='HSI')
    
    LiDAR_data = normalize(LiDAR_data, method=2, modality='LiDAR')
    return generate_data_loaders(HSI_data, LiDAR_data, Train_data, Test_data, GT,
                               batch_size=args.batch_size,
                               window_size=args.window_size)

def initialize_model(args, dataset_info):
  
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张GPU进行训练")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
    model = MultimodalClassifier(
        l1=dataset_info['HSI_Channel'],
        l2=dataset_info['LiDAR_Channel'],
        dim=dataset_info['HSI_Channel'],
        num_classes=dataset_info['num_classes'],
        mode='fusion'
    ).to(device)
    
   
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.003,
        betas=(0.9, 0.999)
    )
    
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    
    
    return model, optimizer, criterion, scheduler, device

def train_epoch(model, train_iter, optimizer, criterion, device):
    model.train()  
    total_loss, correct = 0.0, 0
    for x1, x2, y in train_iter:  
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()  
        y_hat = model(x1, x2) 
        loss = criterion(y_hat, y.long()) 
        loss.backward()  
        optimizer.step()  
        total_loss += loss.item() 
        correct += (y_hat.argmax(dim=-1) == y).float().sum().item()  
    return total_loss / len(train_iter), correct / len(train_iter.dataset) 

def evaluate_model(model, test_iter, device):
    model.eval()  
    y_test, y_pred = [], []
    with torch.no_grad():  
        for x1, x2, y in test_iter:  
            x1, x2, y = x1.to(device), x2.to(device), y.to(device) 
            y_hat = model(x1, x2)  
            y_pred.extend(y_hat.cpu().argmax(dim=1)) 
            y_test.extend(y.cpu()) 
    return np.array(y_test), np.array(y_pred) 

def compute_metrics(y_test, y_pred):
    oa = accuracy_score(y_test, y_pred)  
    confusion = confusion_matrix(y_test, y_pred)  
    each_acc, aa = AA_andEachClassAccuracy(confusion) 
    kappa = cohen_kappa_score(y_test, y_pred) 
    return oa, aa, kappa, confusion

def save_results(args, dataset_info, y_test, y_pred, best_accuracy, aa, kappa, training_time, model_stats, epoch_times):
    classification = classification_report(y_test, y_pred, 
                                        target_names=dataset_info['target_names'], 
                                        digits=4)  
    confusion = confusion_matrix(y_test, y_pred) 

    file_name = os.path.join(args.result_path, args.dataset, f"{args.dataset}.txt")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)  

    with open(file_name, 'a') as f:  
        f.write(f"\n{'*' * 90}\n")
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        
        
        f.write("\n\n=== 超参数配置 ===\n")
        f.write(f"数据集: {args.dataset}\n")
        f.write(f"窗口大小: {args.window_size}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"随机种子: {args.seed}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"权重衰减: {args.weight_decay}\n")
        f.write(f"训练设备: {args.device}\n") 
        
     
        f.write("\n=== 数据集信息 ===\n")
        f.write(f"类别数: {dataset_info['num_classes']}\n")
        f.write(f"HSI通道数: {dataset_info['HSI_Channel']}\n")
        f.write(f"LiDAR通道数: {dataset_info['LiDAR_Channel']}\n")
        

        
        f.write("\n=== 实验结果 ===\n")
        f.write(f'Overall Accuracy (OA): {best_accuracy * 100:.2f}%\n')
        f.write(f'Average Accuracy (AA): {aa * 100:.2f}%\n')
        f.write(f'Kappa Accuracy: {kappa * 100:.2f}%\n')
        f.write(f"\n{classification}\n")
        f.write(f"混淆矩阵:\n{confusion}\n")

        f.write(f"{'*' * 90}\n")

def get_model_attr(model, attr_name):
    """获取模型属性的辅助函数"""
    if isinstance(model, torch.nn.DataParallel):
        return getattr(model.module, attr_name)
    return getattr(model, attr_name)

def plot_fusion_weights(args):
    """绘制融合权重随训练轮数的变化曲线"""
    # 读取权重数据
    weight_file = os.path.join(args.result_path, f"{args.dataset}_weights.txt")
    weights_df = pd.read_csv(weight_file)
    
    # 设置图形大小和DPI
    plt.figure(figsize=(8, 6), dpi=1200)
    plt.style.use('seaborn-whitegrid')
    
    # 绘制散点和虚线
    plt.plot(weights_df['Epoch'], weights_df['Weight1'], 
            color='#1f77b4', label='early fusion',
            linestyle='--',         
            marker='o',              
            markersize=2,           
            markerfacecolor='white', 
            markeredgewidth=1.5)     
    
    plt.plot(weights_df['Epoch'], weights_df['Weight2'], 
            color='#2ca02c', label='middle fusion',
            linestyle='--',
            marker='s',              # 方形标记
            markersize=2,
            markerfacecolor='white',
            markeredgewidth=1.5)
    
    plt.plot(weights_df['Epoch'], weights_df['Weight3'], 
            color='#ff7f0e', label='late fusion',
            linestyle='--',
            marker='^',              # 三角形标记
            markersize=2,
            markerfacecolor='white',
            markeredgewidth=1.5)
    
    # 设置坐标轴范围和刻度
    plt.xlim(1, 60)
    plt.xticks(np.arange(1, 61, 5))  # 主刻度
    plt.gca().set_xticks(np.arange(1, 61, 1), minor=True)  # 小刻度
    
    plt.xlabel('Epochs', fontsize=12, fontfamily='serif')
    plt.ylabel('Weights', fontsize=12, fontfamily='serif')
    
 
    plt.title(f'Fusion Weights Evolution ({args.dataset})', 
             fontsize=14, fontfamily='serif', pad=15)
    
 
    plt.legend(fontsize=10, prop={'family': 'serif'}, 
              frameon=True, fancybox=False, edgecolor='black',
              bbox_to_anchor=(1.02, 1), loc='upper left')
    

    plt.grid(True, linestyle='--', alpha=0.3)
    
   
    plt.xticks(fontfamily='serif', fontsize=10)
    plt.yticks(fontfamily='serif', fontsize=10)

    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    
   
    plt.tight_layout()
    save_path = os.path.join(args.result_path, f"{args.dataset}_weights.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n权重变化曲线已保存至: {save_path}")

def main(args):
    dataset_info = get_dataset_info(args.dataset) 
    if dataset_info is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}") 

    setup_seed(args.seed)  
    os.makedirs(args.model_path, exist_ok=True)  
    os.makedirs(args.result_path, exist_ok=True)  

 
    TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, train_iter, test_iter = prepare_data(args)
    print(f"Dataset: {args.dataset}")
    print(f"TRAIN_SIZE: {TRAIN_SIZE}, TEST_SIZE: {TEST_SIZE}, TOTAL_SIZE: {TOTAL_SIZE}")
    print(f"Training on {args.device}\n")


    model, optimizer, criterion, scheduler, device = initialize_model(args, dataset_info)
    best_accuracy, best_y_pred, best_y_test = 0.0, None, None
    best_train_acc = 0.0
    min_train_loss = float('inf')  
    start_time = time.time()
    epoch_times = [] 


    weight_file = os.path.join(args.result_path, f"{args.dataset}_weights.txt")
    with open(weight_file, 'w') as f:
        f.write("Epoch,Weight1,Weight2,Weight3\n")

    for epoch in range(args.epochs):
        epoch_start_time = time.time()  
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练循环
        for batch_idx, (x1, x2, labels) in enumerate(train_iter):
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(x1, x2)
            
            loss = criterion(outputs, labels)
            
            l2_loss = 0
            for param in model.parameters():
                l2_loss += torch.norm(param)
            loss += args.weight_decay * l2_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if args.multi_gpu:
                torch.cuda.synchronize()  # 确保所有GPU完成计算
        
        epoch_loss = running_loss / len(train_iter)
        epoch_acc = correct / total
        
       
        if args.multi_gpu:
            weights = F.softmax(get_model_attr(model, 'fusion_weights'), dim=0)
        else:
            weights = F.softmax(model.fusion_weights, dim=0)
        
        w1, w2, w3 = weights.detach().cpu().numpy()
        with open(weight_file, 'a') as f:
            f.write(f"{epoch+1},{w1:.4f},{w2:.4f},{w3:.4f}\n")
        
     
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
      
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.6f}, Accuracy: {epoch_acc:.4f}, '
              f'weight1: {w1:.4f}, weight2: {w2:.4f}, weight3: {w3:.4f}, '
              f'Time: {epoch_time:.2f}s')
        
    
        need_test = False
        if (epoch_loss < min_train_loss):  
            best_train_acc = max(epoch_acc, best_train_acc)   
            min_train_loss = min(epoch_loss, min_train_loss)  
            need_test = True

     
        if need_test:
            print('\nTesting...\n')
            y_test, y_pred = evaluate_model(model, test_iter, device)
            oa, aa, kappa, _ = compute_metrics(y_test, y_pred)
            
        
            if oa > best_accuracy:
                best_accuracy = oa
                best_y_pred = y_pred
                best_y_test = y_test
                torch.save(model.state_dict(), os.path.join(args.model_path, f"{args.dataset}.pt"))
                print('Model saved.')
            
            print(f"OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}, Best Accuracy: {best_accuracy:.4f}")
        
        scheduler.step()

   
    print('\n进行最终测试评估...\n')
    y_test, y_pred = evaluate_model(model, test_iter, device)
    oa, aa, kappa, _ = compute_metrics(y_test, y_pred)
 
    if oa > best_accuracy:
        best_accuracy = oa
        best_y_pred = y_pred
        best_y_test = y_test
        torch.save(model.state_dict(), os.path.join(args.model_path, f"{args.dataset}.pt"))
        print('保存最终模型。')

    training_time = time.time() - start_time
    save_results(args, dataset_info, best_y_test, best_y_pred, best_accuracy, aa, kappa, training_time, model_stats, epoch_times)
    plot_fusion_weights(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态遥感图像分类训练脚本") # 创建参数解析器
    
    # 数据集参数
    parser.add_argument("--dataset", type=str, default="Houston",
                        choices=['Houston', 'MUUFL', 'Trento', 'Berlin', 'Augsburg', 'Houston2018'],
                        help="要使用的数据集")
    

    parser.add_argument("--window_size", type=int, default=9,
                        help="窗口大小")
    
 
    parser.add_argument("--epochs", type=int, default=60,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="批次大小")
    parser.add_argument("--seed", type=int, default=100,
                        help="随机种子")
    
  
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                        help="权重衰减")

   
    parser.add_argument("--model_path", type=str, default="./models/",
                        help="模型保存路径")
    parser.add_argument("--result_path", type=str, default="./results/",
                        help="结果保存路径")
    
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="训练设备")
    parser.add_argument("--gpu", type=int, default=0,
                        help="选择使用的GPU编号 (0, 1, 2, ...)")
    
    parser.add_argument("--multi_gpu", action="store_true",
                        help="是否使用多GPU训练")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3",
                        help="要使用的GPU ID列表，用逗号分隔")
    
    args = parser.parse_args() 
    

    if args.multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.batch_size = args.batch_size * torch.cuda.device_count()
        print(f"多GPU训练模式，调整后的批次大小: {args.batch_size}")
    else:
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    
    main(args)  