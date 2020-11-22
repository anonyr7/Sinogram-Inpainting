Please download the TCIA dataset from: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026#bcab02c187174a288dbcbf95d26179e8

Or the augmented training and testing dataset from (Recommended): https://drive.google.com/drive/folders/1vqjFQl_gZ3yAm1_XYDUmGtehqdL-ASL9?usp=sharing

Unzip the data and put in the current folder.


### Data Augmentation

We performed the following random affine transformation on the raw data.

```
dataset = dset.ImageFolder(root=folder,
                            transform=transforms.Compose([
                                transforms.Grayscale(),
                                transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.5,1.1), shear=(-20,20,-20,20)),
                                transforms.ToTensor(),
                            ]))
```