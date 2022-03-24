# 2-Step Sparse-View CT Reconstruction with a Domain-Specific Perceptual Network

### Dependencies:

- python=3.7
- pytorch=1.6.0
- torch-radon=0.0.1 (https://github.com/matteo-ronchetti/torch-radon)

Some models such as FISTA-TV require another environment, specified in its document. 

### Instructions:

1. Run the commands indicated in each model's folder to train or test the models with given checkpoint files and toy datasets (example test images that are excluded from the training set).
    - Train/test SIN, and then train/test 4c-PRN. Intermediate results are saved in Toy-Dataset.

2. Evaluate the predicted reconstruction images with `Compute_Metrics.ipynb`.

3. For dataset download please refer to further instructions in the `Data` folder. The jupyter notebook `TCIA_data_preprocessing.ipynb` includes our way of data augmentation.

4. We provide our implementation of the state-of-the-art models mentioned in the paper: FISTA-TV, cGAN, and Neumann Networks in `State-of-the-art` folder.

### Cone-Beam Example Results:

This is a preliminary experiment for extension to 3D cone-beam reconstructions, the walnut data come from https://github.com/cicwi/WalnutReconstructionCodes.

Sparse-View FDK | SIN | SIN-4c-PRN    |  Ground Truth
:--------------:|:---:|:------------:|:---------------:
<img src="https://github.com/anonyr7/Sinogram-Inpainting/blob/master/Toy-Dataset/Walnut6_sparseFDK-min.gif?raw=true" width="200" height="200"/>| <img src="https://github.com/anonyr7/Sinogram-Inpainting/blob/master/Toy-Dataset/Walnut6_SIN-min.gif?raw=true" width="200" height="200"/>|  <img src="https://github.com/anonyr7/Sinogram-Inpainting/blob/master/Toy-Dataset/Walnut6_SIN-4c-PRN-min.gif?raw=true" width="200" height="200"/>|  <img src="https://github.com/anonyr7/Sinogram-Inpainting/blob/master/Toy-Dataset/Walnut6_GT-min.gif?raw=true" width="200" height="200"/>


Please cite our paper:
```
@article{wei20202,
  title={2-step sparse-view ct reconstruction with a domain-specific perceptual network},
  author={Wei, Haoyu and Schiffers, Florian and W{\"u}rfl, Tobias and Shen, Daming and Kim, Daniel and Katsaggelos, Aggelos K and Cossairt, Oliver},
  journal={arXiv preprint arXiv:2012.04743},
  year={2020}
}
```
