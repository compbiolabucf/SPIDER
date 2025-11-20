# **SPIDER: Spatially Integrated Denoising via Embedding Regularization with Single-Cell Supervision**

<p align="center">
  <img src="Model.png" width="700">
</p>

**Figure:** *Overview of SPIDER framework.* SPIDER integrates spatial transcriptomics (ST) data with reference scRNA-seq information by jointly leveraging spatial structure, transcriptional similarity, and pseudo-ST supervision. The framework first constructs three graphs: a spatial neighborhood graph derived from spot coordinates, and two transcriptional similarity graphs (TSGs) built from real ST spots and pseudo-spots generated from scRNA-seq data from the same tissue type. A tri-branch graph attention encoder then learns low-dimensional latent representations from these graphs. The pseudo-ST branch incorporates supervision by optimizing cell-type ratio prediction, while an embedding regularization term aligns the real ST and pseudo-ST embeddings to transfer cell-type structure from the reference domain. A unified graph attention decoder reconstructs gene expression using a zero-inflated negative binomial likelihood. The resulting denoised expression matrix can be used for downstream analyses, including spatial domain identification, marker gene pattern recovery, and improved clustering performance.*

---

## 📦 Installation

```bash

conda create -n spider python=3.8.20
conda activate spider
git clone https://github.com/compbiolabucf/SPIDER.git
cd SPIDER
pip install -r requirements.txt
python setup.py build
python setup.py install

```
