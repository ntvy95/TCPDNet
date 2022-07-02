# Two-Step Color-Polarization Demosaicking Network

This is the source code github repository for our paper "Two-Step Color-Polarization Demosaicking Network".

## System requirements

- Pytorch 1.8.0.
- CUDA 11.0. In case you run into troubles of setting up the appropriate CUDA toolkit version, please refer to [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).
- GPU 6GB at minimum. A high GFLOPS for FP32 operations is preferred as our network parameters are stored in 32-bit format. 

## Environment setup

Setup the anaconda environment as follows:

```
conda env create --name <environment_name> -f conda_environment.yml
```

Replace `<environment_name>` with any names you would like to.

## Dataset preparation

### Tokyo Tech dataset

Download the data from [this link](http://www.ok.sc.e.titech.ac.jp/res/PolarDem/data/Dataset.zip). Move its extracted folder `mat`to `dataset/TokyoTech/` and then run the script `dataset/TokyoTech/generate_hdf5.py` to convert the data to hdf5 format.

### CPDNet dataset

Download the data from [this link](https://github.com/wsj890411/CPDNet). Move its extracted folder `dataset` to `dataset/OL/` and then run the script `dataset/OL/generate_hdf5.py` to convert the data to hdf5 format.

## Evaluation

### Bilinear interpolation

Tokyo Tech dataset:

```
python evals.py --config configs/bilinear_1.yaml --mode best_eval
```

CPDNet dataset:

```
python evals.py --config configs/bilinear_1_OL.yaml --mode best_eval
```

### Single-step network & TCPDNet

```
python evals.py --config <config_path> --mode avg_5_models
```

| In paper | Evaluation dataset | <config_path> |
| --- | --- | --- |
| TCPDNet (<img src="https://render.githubusercontent.com/render/math?math=L_{C}\!%2B4L_{CP}^{YCbCr}">) | Tokyo Tech | configs/unet_sim_ycbr_4.yaml |
| TCPDNet (<img src="https://render.githubusercontent.com/render/math?math=L_{C}\!%2B4L_{CP}^{YCbCr}">) | CPDNet | configs/unet_sim_OL_ycbr_4.yaml |
| TCPDNet (<img src="https://render.githubusercontent.com/render/math?math=L_{C}\!%2B4L_{CP}">) | Tokyo Tech | configs/unet_sim_rgb_4.yaml |
| Single-step network (<img src="https://render.githubusercontent.com/render/math?math=L_{CP}">) | Tokyo Tech | configs/unet_e2e.yaml |
| Single-step network (<img src="https://render.githubusercontent.com/render/math?math=L_{CP}^{YCbCr}">) | Tokyo Tech | configs/unet_e2e_ycbr.yaml |


A note regarding the reported result of the re-trained CPDNet in our paper: We re-trained CPDNet according to our training protocol rather than following the protocol indicated by its original code for a fair comparison.

## Training

```
python train.py --config <config_path>
```

## Reference

The code is available only for research purpose. If you use this code for publications, please cite the following paper.

"Two-Step Color-Polarization Demosaicking Network"  
Vy Nguyen, Masayuki Tanaka, Yusuke Monno, and Masatoshi Okutomi  
Proceedings of IEEE International Conference on Image Processing (ICIP2022), October 2022 (to appear)
