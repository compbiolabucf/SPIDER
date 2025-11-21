pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-geometric==2.4.0
pip install rpy2==3.4.1

Rscript -e 'install.packages(
    "https://cran.r-project.org/src/contrib/Archive/mclust/mclust_5.4.10.tar.gz",
    repos = NULL,
    type = "source"
)'