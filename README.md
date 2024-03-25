# Diff-BGM: A Diffusion Model for Video Background Music Generation

Official implementation for CVPR 2024 paper: **Diff-BGM: A Diffusion Model for Video Background Music Generation**

By Sizhe Li, Yiming Qin, Minghang Zheng, Xin Jin, Yang Liu.

- Thanks for the code structure from [Polyffusion](https://github.com/aik2mlj/polyffusion/tree/sdf_prmat2c%2Bpop909)

## 1. Installation

``` shell
pip install -r requirements.txt
pip install -e polyffusion
pip isntall -e polyffusion/mir_eval
```

## 2. Training

### Preparations

1. The extracted features of the dataset POP909 can be accessed [here](https://yukisaki-my.sharepoint.com/personal/aik2_yukisaki_io/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Faik2%5Fyukisaki%5Fio%2FDocuments%2FShare%2Fpolyffusion%2FPOP909%5F4%5Fbin%5Fpnt%5F8bar%2Ezip&parent=%2Fpersonal%2Faik2%5Fyukisaki%5Fio%2FDocuments%2FShare%2Fpolyffusion&ga=1). Please put it under `/data/` after extraction.

2. The extracted features of the dataset BGM909 can be accessed [here](https://drive.google.com/drive/folders/1zRNROuTxVNhJfqeyqRzPoIY60z5zLaHK?usp=drive_link). Please put them under `/data/bgm909/` after extraction.

3. The needed pre-trained models for training can be accessed [here](https://yukisaki-my.sharepoint.com/personal/aik2_yukisaki_io/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Faik2%5Fyukisaki%5Fio%2FDocuments%2FShare%2Fpolyffusion%2Fpretrained%5Fmodels%5Ffor%5Fpolyffusion%2Ezip&parent=%2Fpersonal%2Faik2%5Fyukisaki%5Fio%2FDocuments%2FShare%2Fpolyffusion&ga=1). Please put them under `/pretrained/` after extraction.

### Commands

```
python polyffusion/main.py --model ldm_chd8bar --output_dir [output_dir]
```

## 3. Inference

Please use the following message to generate music for videos in BGM909.

```
python polyffusion/inference_sdf.py --model_dir=[model_dir] --uncond_scale=5.
```

## 4. Test

To reproduce the metrics in our original paper, please refer to `/polyffusion/test.ipynb`.

We provide the generation results [here](https://drive.google.com/drive/folders/1kYQLAmw8-zyBx43RW7aUSE8VXcFDxkez?usp=sharing).