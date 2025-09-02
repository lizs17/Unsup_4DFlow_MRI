# Unsup_4DFlow_MRI
The implementation for the "Unsupervised 4D-flow MRI reconstruction based on partially-independent generative modeling and complex-difference sparsity constraint".

Generator network:
![Image text](figures/Network.png)

Optimization algorithm:
![Image text](figures/Algorithm.png)

# Python Environment

* python=3.8.18
* numpy=1.24.3
* pytorch=1.12.1

The supporting file `requirements.yml` is provided for a quick environment configuration using anaconda.

# Contents

This repository contains the source code for unsupervised 4D-flow MRI reconstruction:

```bash
├── data/
│   ├── [dataName].h5
├── mask/
│   ├── [maskName].h5
├── utils.py
├── utils_4DFlow.py
├── model.py
├── pretrain.py
├── finetune.py
```
* `data/`: the 4D-flow MRI data directory. Each data is saved in .h5 format, which contains a fully-sampled image 'imgGT', the coil sensitivity maps 'smaps', and the segmented binary mask for blood vessels 'maskROI'.
* `mask/`: the k-space sampling pattern directory, containing the undersampling mask in the ky-kz-time domain.
* `utils.py`: the helper functions and classes
* `utils_4DFlow.py`: the helper functions for flow quantification
* `model.py`: the partially-independent generator architecture
* `pretrain.py`: the code for pretraining the generator
* `finetune.py`: the code of the ADMM algorithm for finetuning the reconstructed images



