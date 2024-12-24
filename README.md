# CBT

A new Cross Band Transformer (CBT) and a Wavelet Cross Band Transformer (Wav-CBT) architecture for pansharpening of satellite imagery

# Performance on GaoFen-2

![alt text](https://github.com/nickdndndn/CBT/blob/main/Images/qualitative.png?raw=true)

# Datasets

The GaoFen-2 and WorldView-3 dataset download links can be found [here](https://github.com/liangjiandeng/PanCollection)
The Sev2Mod dataset can be download [here](https://zenodo.org/records/8360458)

# List of benchmark methods implemented in this study

 Implementation and pretrained weights of benchmark methods on GaoFen-2 and WorldView3 datasets.
 
- [PNN](https://github.com/VisionVoyagerX/PNN)
- [PanNet](https://github.com/VisionVoyagerX/PanNet)
- [GPPNN](https://github.com/VisionVoyagerX/GPPNN)
- [MSDCNN](https://github.com/VisionVoyagerX/MDCUN)
- [BiMPan](https://github.com/VisionVoyagerX/BiMPan)
- [PanFormer](https://github.com/VisionVoyagerX/PanFormer)
- [ArbRPN](https://github.com/VisionVoyagerX/ArbRPN)

# Project Setup

This project requires downloading datasets for training and testing purposes. Follow the steps below to set up the project:

## Step 1: Clone Repository

Clone the project repository to your local machine:

```
git clone https://github.com/VisionVoyagerX/CBT.git && cd CBT
```

## Step 2: Download Datasets and Organize

Download and extract the datasets, then organize them according to the specified file structure below. Ensure the file is placed in the root directory of the CBT project.

- GF2
    - train
    - val
    - test
- WV3
    - train
    - val
    - test
- SEV2MOD
    - train
    - val
    - test

## Step 3: Train (optional)

`
python3 train.py -c [choose config from /configs file].yaml
`

## Step 4: Inference

`
python3 inference.py -c [choose config from /configs file].yaml
`


