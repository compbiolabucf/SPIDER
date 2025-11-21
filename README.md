![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
# **SPIDER: Spatially Integrated Denoising via Embedding Regularization with Single-Cell Supervision**

<p align="center">
  <img src="resources/Model.png" width="700">
</p>

**Figure:** *Overview of SPIDER framework.* SPIDER integrates spatial transcriptomics (ST) data with reference scRNA-seq information by jointly leveraging spatial structure, transcriptional similarity, and pseudo-ST supervision. The framework first constructs three graphs: a spatial neighborhood graph derived from spot coordinates, and two transcriptional similarity graphs (TSGs) built from real ST spots and pseudo-spots generated from scRNA-seq data from the same tissue type. A tri-branch graph attention encoder then learns low-dimensional latent representations from these graphs. The pseudo-ST branch incorporates supervision by optimizing cell-type ratio prediction, while an embedding regularization term aligns the real ST and pseudo-ST embeddings to transfer cell-type structure from the reference domain. A unified graph attention decoder reconstructs gene expression using a zero-inflated negative binomial likelihood. The resulting denoised expression matrix can be used for downstream analyses, including spatial domain identification, marker gene pattern recovery, and improved clustering performance.*

---

## 📦 Installation

Install R 4.5.1

You can try using the command

```bash
sudo apt update && sudo apt install -y build-essential gfortran libreadline-dev libx11-dev libxt-dev libpng-dev libjpeg-dev libcairo2-dev libssl-dev libcurl4-openssl-dev libbz2-dev liblzma-dev libpcre2-dev libxml2-dev zlib1g-dev wget tar && \
wget https://cran.r-project.org/src/base/R-4/R-4.5.2.tar.gz && \
tar -xvf R-4.5.2.tar.gz && cd R-4.5.2 && \
./configure --enable-R-shlib --with-blas --with-lapack && \
make -j$(nproc) && sudo make install && R --version
```

Install the conda environment using the bash file 
```bash
chmod +x create_environment.sh
./create_environment.sh
conda activate spider
```

Install SPIDER inside the environment 
```bash
git clone https://github.com/compbiolabucf/SPIDER.git
cd SPIDER
python setup.py build
python setup.py install

```



## Running Denoising 

The sample dataset can be downloaded from this [link](https://drive.google.com/file/d/14sieoleV-a8Hx9KVVWDFZk5-5MjGsRLT/view?usp=sharing).

The sample dataset is created using ***preprocess_data.py***, using the DLPFC ST and SC data to create pseudo ST data. 

The file contains preprocessed DLPFC data and a pseudo spot data created from reference scRNA-seq data. 
Use the ***tutorial.ipynb*** file to run the denoiser. 



## Dependencies

The SPIDER framework requires the following core dependencies:

```
python = 3.8.20
torch == 2.1.0+cu118
torch-geometric == 2.4.0
torch-cluster == 1.6.3+pt21cu118
torch-scatter == 2.1.2
torch-sparse == 0.6.18
R == 4.5.1 (R package mclust 5.4.10)
```
## Citation
