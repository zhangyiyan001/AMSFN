## INNFUS_AMSFN
The official implementation for "**Adaptive Multi-Stage Fusion of Hyperspectral and LiDAR Data via Selective State Space Models**", Information Fusion, 2025.
![AMSFN](https://github.com/zhangyiyan001/AMSFN/blob/main/AMSFN_framework.png)
****

### Requirements
```bash
torch>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
pandas>=1.3.0
pyyaml>=5.4.0
einops>=0.4.0
timm>=0.6.0
fvcore>=0.1.5
```

## How to run it
```
python main.py --dataset Houston --window_size 13 --lr 0.0005 --gpu 0 --epoch 60
```

## Citation
If you find our work helpful in your research, kindly consider citing it. We appreciate your support！
```
@article{zhang2025adaptive,
  title={Adaptive multi-stage fusion of hyperspectral and LiDAR data via selective state space models},
  author={Zhang, Yiyan and Gao, Hongmin and Chen, Zhonghao and Fei, Shuyu and Zhou, Jun and Ghamisi, Pedram and Zhang, Bing},
  journal={Information Fusion},
  pages={103488},
  year={2025},
  publisher={Elsevier}
}
```
