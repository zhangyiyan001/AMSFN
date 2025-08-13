from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mamba.mambablock import CrossMambaFusionBlock, ConcatMambaFusionBlock
from local_vmamba import VSSBlock
import torch.nn.functional as F

def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1) -> nn.Sequential:
    """创建一个卷积-批归一化-ReLU激活的组合层"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-3),
        nn.ReLU()
    )

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""
    def __init__(self, dim: int):
        super().__init__()
        # 1. Q,K,V转换层
        self.q_conv = nn.Conv2d(dim, dim, 1) 
        self.k_conv = nn.Conv2d(dim, dim, 1)
        self.v_conv = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5 
        
        # 2. 输出投影层
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        
        # 3. 自适应门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, 1, 1), 
            nn.Sigmoid()  
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x1.shape 
        
        # 1. 特征转换 (使用1x1卷积进行线性变换，不改变通道数)
        q = self.q_conv(x1).view(B, C, -1)     
        k = self.k_conv(x2).view(B, C, -1)     
        v = self.v_conv(x2).view(B, C, -1)    
        
        # 2. 计算注意力权重
        attn = (q.transpose(-2, -1) @ k) * self.scale 
        attn = F.softmax(attn, dim=-1)  
        
        # 3. 注意力加权 (将注意力权重应用到value上)
        fusion = (v @ attn.transpose(-2, -1))    
        fusion = fusion.view(B, C, H, W)         
        fusion = self.proj(fusion)               
        
        # 4. 门控融合 (计算自适应门控权重，控制融合特征的比例)
        gate = self.gate(torch.cat([x1, x2], dim=1))  
        
        # 5. 残差连接 (将原始输入x1与融合特征进行加权求和)
        return x1 + gate * fusion  # 最终输出 = 原始输入 + 门控权重 * 融合特征



class MultiScaleFusion(nn.Module):
   
    def __init__(self, dim: int):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(2**i),
                nn.Conv2d(dim, dim, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            ) for i in range(3)
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度特征提取
        features = [scale(x) for scale in self.scales]
        # 上采样对齐
        aligned_features = [F.interpolate(f, size=x.shape[-2:], 
                                        mode='bilinear', 
                                        align_corners=True)  
                          for f in features]
        # 特征融合
        return self.fusion(torch.cat(aligned_features, dim=1))



class FeatureAggregation(nn.Module):
    def __init__(self, dim: int, num_features: int):
        super().__init__()
        self.attention_hsi = nn.Sequential(
            nn.Linear(dim, num_features),
            nn.Softmax(dim=-1)
        )
        self.attention_lidar = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1), 
            nn.Sigmoid()  
        )
        
   
        self.modal_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1), 
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, 2, 1),  
            nn.Softmax(dim=1)  
        )

    def forward(self, features_hsi: List[torch.Tensor], features_lidar: List[torch.Tensor]) -> torch.Tensor:
        B, C, H, W = features_hsi[0].shape
        
        # 处理HSI特征
        hsi_vectors = [f.mean([2, 3]) for f in features_hsi]  # [B, C]
        weights_hsi = self.attention_hsi(hsi_vectors[0])  # [B, num_features]
        fused_hsi = sum(w.view(B, 1, 1, 1) * f 
                       for w, f in zip(weights_hsi.transpose(0,1), features_hsi))
        
        # 处理LiDAR特征
        fused_lidar = torch.zeros_like(features_lidar[0])
        for feature in features_lidar:
            # 生成空间注意力图 [B, 1, H, W]
            spatial_attention = self.attention_lidar(feature)
            # 应用空间注意力
            fused_lidar += feature * spatial_attention
        
        # 保持空间维度的特征融合
        modal_features = torch.cat([fused_hsi, fused_lidar], dim=1)  # [B, C*2, H, W]
        modal_weights = self.modal_fusion(modal_features)  # [B, 2, H, W]
        
        # 空间自适应的模态融合
        final_output = (modal_weights[:, 0:1, :, :] * fused_hsi + 
                       modal_weights[:, 1:2, :, :] * fused_lidar)
        
        return final_output


class MultimodalClassifier(nn.Module):

    def __init__(self, l1: int, l2: int, dim: int, num_classes: int, mode: str = 'fusion'):
        
        super(MultimodalClassifier, self).__init__()
        self.mode = mode
        
    
        self.conv_lidar = conv_bn_relu(l2, l1, kernel_size=1, stride=1, padding=0)
        
      
        self.early_fusion = nn.Sequential(
            nn.Conv2d(l1*2, l1, 1),
            nn.BatchNorm2d(l1),
            nn.ReLU()
        )
        
     
        self.vssblocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim,
                    drop_path=0.2,
                    d_state=16,
                    direction=['h', 'v', 'w2', 'w2_flip'])
            for _ in range(4)
        ])
        
     
        self.fusion_blocks = nn.ModuleList([
            CrossAttentionFusion(dim) 
            for _ in range(4)
        ])
        
     
        self.multi_scale_fusion = MultiScaleFusion(dim)
        
      
        self.concat_mamba_fusion = ConcatMambaFusionBlock(
            dim=dim,
            drop_path=0.2,
            mlp_ratio=2.0,
            d_state=16
        )
        
        # 分类头
        self.classifier = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', nn.Flatten(1))
        ]))
        
        self.linear = nn.Linear(dim, num_classes, bias=False)
        
   
        weights = torch.rand(3)
        weights = weights / weights.sum() 
        self.fusion_weights = nn.Parameter(weights)
        
     
        self.feature_aggregation = FeatureAggregation(dim, 4)
        
       
        self.dropout = nn.Dropout(0.2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None, return_weights: bool = False):
        if self.mode == 'hsi':
            
            x = x1
            features = []
            for vssblock in self.vssblocks:
                x = vssblock(x)
                features.append(x)
            
            x = self.multi_scale_fusion(x)
            x = self.dropout(x)
            x = self.classifier(x)
            x = self.dropout(x)
            out = self.linear(x)
            return out
            
        elif self.mode == 'lidar':
         
            x = self.conv_lidar(x2)  
            features = []
            for vssblock in self.vssblocks:
                x = vssblock(x) 
                features.append(x)
            
            x = self.multi_scale_fusion(x)
            x = self.dropout(x)
            x = self.classifier(x)
            x = self.dropout(x)
            out = self.linear(x)
            return out
            
        else:
            
            x2 = self.conv_lidar(x2)
            
           
            early_features = self.early_fusion(torch.cat([x1, x2], dim=1)) 
            early_features = self.multi_scale_fusion(early_features)
            early_features = self.dropout(early_features)
            
           
            x1_mid, x2_mid = x1, x2
            middle_features_hsi = []
            middle_features_lidar = []
            for vssblock, fusion_block in zip(self.vssblocks, self.fusion_blocks):
                x1_mid = vssblock(x1_mid)
                x2_mid = vssblock(x2_mid)
                middle_features_hsi.append(x1_mid)
                middle_features_lidar.append(x2_mid)
            
            middle_fusion = self.feature_aggregation(middle_features_hsi, middle_features_lidar)
            middle_fusion = self.multi_scale_fusion(middle_fusion)
            middle_fusion = self.dropout(middle_fusion)
            
          
            x_late = self._apply_concat_mamba(x1_mid, x2_mid)
            x_late = self.multi_scale_fusion(x_late)
            x_late = self.dropout(x_late)
            
          
            weights = F.softmax(self.fusion_weights, dim=0)
            final_features = (weights[0] * early_features + 
                            weights[1] * middle_fusion + 
                            weights[2] * x_late)
            
          
            out = self.classifier(final_features)
            out = self.dropout(out)
            out = self.linear(out)
            
            if return_weights:
                return out, (weights[0].item(), weights[1].item(), weights[2].item())
            return out

    def _apply_concat_mamba(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """连接Mamba融合"""
        x_fusion = self.concat_mamba_fusion(x1.permute(0, 2, 3, 1).contiguous(),
                                          x2.permute(0, 2, 3, 1).contiguous())
        return x_fusion.permute(0, 3, 1, 2).contiguous()





