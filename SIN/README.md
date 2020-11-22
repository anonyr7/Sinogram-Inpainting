# SIN Model

### Train:

Run command `python3 main.py -h` to see the argument options and information.

Example:

```
CUDA_VISIBLE_DEVICES=0 python3 main.py \
--epochs 100 --batchSize 20 -l 0.0001 \
--datadir ../Data/sinograms/train \
--outdir out \
--load -1
```

### Test Single Image:

Run command `python3 predict_single.py -h` to see the argument options and information.

Example:

```
CUDA_VISIBLE_DEVICES=0 python3 predict_single.py \
--testImage ../Toy-Dataset/Ground_Truth_sinogram/C/47.png \
--ckpt out/ckpt/G_epoch49.pth
```

Output is in the current folder by default.

### Test All Images from folder:

Run command `python3 predict_all.py -h` to see the argument options and information.

Example:

```
CUDA_VISIBLE_DEVICES=0 python3 predict_all.py \
--datadir ../Toy-Dataset/Ground_Truth_sinogram \
--outdir ../Toy-Dataset/SIN \
--ckpt out/ckpt/G_epoch49.pth \
--class_name N
```