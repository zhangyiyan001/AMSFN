## INFFUS AMSFN
This is the official implementation for the paper **"Adaptive Multi-Stage Fusion of Hyperspectral and LiDAR Data via Selective State Space Models"**, published in *Information Fusion*, 2025.
![AMSFN](https://github.com/zhangyiyan001/AMSFN/blob/main/framework.png)
****


## Environment

We have successfully tested the environment only on Linux. Please ensure you have the appropriate versions of PyTorch and CUDA installed that match your computational resources.

1.  **Create and activate the Conda environment:**
    ```shell
    conda create -n AMSFN python=3.9
    conda activate AMSFN
    ```

2.  **Install dependencies:**
    First, install PyTorch, CUDA, and cuDNN. Then, install the remaining dependencies with the following command:
    ```shell
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    ```

3.  **Install Mamba:**
    ```shell
    cd mamba/selective_scan && pip install . && cd ../../
    ```

## Dataset

The datasets used in this project can be downloaded from Baidu Netdisk.

Link: https://pan.baidu.com/s/12bPrqTl3Iqbfs7jgd5fgqA
Extraction Code: k95x


## Usage

You can run the model on different datasets using the following commands:

**For the Houston Dataset:**
```shell
python main.py --dataset Houston --window_size 13 --lr 0.0005 --gpu 0
```

**For the Trento Dataset:**
```shell
python main.py --dataset Trento --window_size 9 --lr 0.0005 --gpu 0
```

**For the MUUFL Dataset:**
```shell
python main.py --dataset MUUFL --window_size 5 --lr 0.0008 --gpu 0
```

## Acknowledgements

Our code is based on the following outstanding projects. We thank the original authors for making their work public!

*   [SIGMA](https://github.com/zifuwan/Sigma)
*   [LocalMamba](https://github.com/hunto/LocalMamba)


## Citation

If you find our work helpful in your research, we would appreciate it if you would consider citing our paper. Your support is our greatest motivation!

> ```bibtex
> @article{zhang2025adaptive,
>   title={Adaptive multi-stage fusion of hyperspectral and LiDAR data via selective state space models},
>   author={Zhang, Yiyan and Gao, Hongmin and Chen, Zhonghao and Fei, Shuyu and Zhou, Jun and Ghamisi, Pedram and Zhang, Bing},
>   journal={Information Fusion},
>   pages={103488},
>   year={2025},
>   publisher={Elsevier}
> }
> ```


## Contact

If you have any questions or suggestions, please feel free to reach out.
Email: zhangyiyan@hhu.edu.cn

## License

This project is released under the Apache 2.0 License.



