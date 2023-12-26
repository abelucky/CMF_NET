# CMF_Net
# Cross-contrast Mutual Fusion Network for Joint MRI Reconstruction and Super-Resolution

## Data
We used two public datasets for our experiments: fastMRI and ISLES. For fastMRI, we experimented with knee images in two modalities: PDWI and FS-PDWI. For ISLES, we experimented with brain images in two modalities: T1WI and T2WI.

## Dependencies
* numpy==1.18.5
* scikit_image==0.16.2
* torchvision==0.8.1
* torch==1.7.0
* runstats==1.8.0
* pytorch_lightning==0.8.1
* h5py==2.10.0
* PyYAML==5.4

**Train**
```bash
cd experimental/MINet/
python3 train.py
```
