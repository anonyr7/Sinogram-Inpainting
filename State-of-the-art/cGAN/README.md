# PyTorch Implementation of Learning-Based State-of-the-Art

Paper: https://ieeexplore.ieee.org/abstract/document/8448403

### Train:

Run command `python3 main.py -h` to see the argument options and information.

Example:

```
CUDA_VISIBLE_DEVICES=0 python3 main.py \
--epochs 100 --batchSize 32 -l 0.0001 \
--outdir out \
--datadir ../../Data/sinograms/train \
--load -1
```

### Test Single Image:

Run command `python3 predict_single.py -h` to see the argument options and information.

Example:

```
python3 predict_single.py \
--ckpt out/ckpt/G_epoch110.pth \
--testImage ../../Toy-Dataset/Ground_Truth_sinogram/C/47.png
```

Output is in the current folder by default.

### Test All Images from folder:

Run command `python3 predict_all.py -h` to see the argument options and information.

Example:

```
CUDA_VISIBLE_DEVICES=0 python3 predict_all.py \
--datadir ../../Toy-Dataset/Ground_Truth_sinogram \
--outdir ../../Toy-Dataset/CGAN \
--ckpt out/ckpt/G_epoch110.pth \
--class_name N
```

Reference:

[1] Ghani, Muhammad Usman, and W. Clem Karl. "Deep learning-based sinogram completion for low-dose CT." 2018 IEEE 13th Image, Video, and Multidimensional Signal Processing Workshop (IVMSP). IEEE, 2018.
