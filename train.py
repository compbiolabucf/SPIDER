from SPIDER import train_SPIDER, Cal_knn_expression, Cal_Spatial_Net, mclust_R
import os, pickle, pandas as pd
import scanpy as sc, numpy as np
import  sklearn
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score
from sklearn.metrics.cluster import adjusted_rand_score
import sys


def run_training():

    if len(sys.argv) < 2:
        print("Usage: python train.py <data_path.pkl>")
        sys.exit(1)

    data_path = sys.argv[1]

    if not os.path.exists(data_path):
        print(f"ERROR: File not found: {data_path}")
        sys.exit(1)

    with open(data_path, "rb") as f:
        loaded = pickle.load(f)
    print("Data loaded")

    # If using the provided CID44971 data please modify the code accordingly 
    
    std, psd = loaded["realST"], loaded["pseudoST"]
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    fname = os.path.splitext(os.path.basename(data_path))[0]

    params = {'W_recon': 3.5611961149150226, 'W_mmd': 1.109798809633746, 'W_cell': 1.280970162059623, 'k_cutoff': 9, 'lr': 0.0004790792935102623}
    k_cutoff = params["k_cutoff"]
    rad_cutoff = std.uns['rad_cutoff']
    Cal_knn_expression(std, k_cutoff=k_cutoff)
    Cal_knn_expression(psd, k_cutoff=k_cutoff)
    Cal_Spatial_Net(std, rad_cutoff=rad_cutoff)

    std,model = train_SPIDER(std, psd, key_added='SPIDER',lr=params["lr"],n_epochs=500,device='cuda:0',random_seed=3156,
                                W_recon=params["W_recon"],W_mmd=params["W_mmd"],W_cell=params["W_cell"])
    
    sc.pp.normalize_total(std,layer='SPIDER',target_sum=1e4)
    sc.pp.log1p(std,layer='SPIDER')
    std.obsm['SPIDER_pca'] = sc.tl.pca(std.layers['SPIDER'], n_comps=20, svd_solver="arpack", random_state=3156)
    mclust_R(std, used_obsm='SPIDER_pca',num_cluster=std.uns['num_cluster'],modelNames="EEI",random_seed=3156)

    ARI = adjusted_rand_score(std.obs['mclust'], std.obs['Ground Truth'])
    NMI = normalized_mutual_info_score(std.obs['mclust'], std.obs['Ground Truth'])
    HS = homogeneity_score(std.obs['mclust'], std.obs['Ground Truth'])
    print(f"Clustering metrics ARI:{ARI}, NMI: {NMI}, HS: {HS}")

    fig = sc.pl.embedding(std,basis="spatial", color=["mclust","Ground Truth"],s=100, title=[f"SPIDER\nClustering metrics ARI:{ARI}, NMI: {NMI}, HS: {HS}","Ground Truth"],show=False, return_fig=True)
    fig.savefig(os.path.join(results_dir,f"{fname}_cluster.pdf"), dpi=300)



if __name__ == "__main__":
    run_training()