from SPIDER import pseudo_spot_generation
import os, scanpy as sc, pandas as pd, numpy as np
def load_DLPFC():
    data_name = "DLPFC_sc_st_ps.pickle"  # name for processed data file
    save_directory = "path to directory to save processed data"
    if os.path.isfile(save_directory+data_name):
        with open(save_directory+data_name, "rb") as f:
            loaded = pickle.load(f)
        scadata, adata, pseudo_adata = loaded['sc'],loaded['realST'],loaded['pseudoST']
        adata.X = adata.X.toarray()
        sc.pp.normalize_total(adata,target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.normalize_total(pseudo_adata,target_sum=1e4)
        sc.pp.log1p(pseudo_adata)
        adata.uns['rad_cutoff'] = 239
        adata.uns['num_cluster'] = 7
        scadata.obsm['spatial'] = coord.loc[scadata.obs_names,:][['X','Y']].values.astype(float)
        adata = adata[~adata.obs['Ground Truth'].isna()]
        return scadata, adata, pseudo_adata

    input_dir = os.path.join('path to raw DLPFC data folder')
    adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    stgenes = adata.var_names

    meta = pd.read_csv("path to DLPFC scRNA-seq data/metadata.csv",index_col=0)
    expr = pd.read_csv("path to DLPFC scRNA-seq data/expr.csv.gz",index_col=0,low_memory=False).T
    barcodes = list(expr.index)
    scgenes = list(expr.columns)
    counts = expr.values
    scadata = sc.AnnData(X=counts,var=pd.DataFrame(index=scgenes),obs=pd.DataFrame(index=barcodes))

    common_genes = list(set(scgenes).intersection(set(stgenes)))
    adata = adata[:,common_genes]

    Ann_df = pd.read_csv(os.path.join(input_dir, 'metadata.tsv'), sep='\t')
    Ann_df.rename(columns={"layer_guess_reordered":"Ground Truth"},inplace=True)
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    adata = adata[adata.obs['Ground Truth']!='nan',]
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata = adata[:,adata.var['total_counts']>100]
    sc.pp.highly_variable_genes(adata,n_top_genes=3000,flavor="seurat_v3")
    hvg = list(adata[:,adata.var['highly_variable']].var_names)
    select_genes = list(set(hvg))
    adata = adata[:,select_genes]
    adata = adata[~adata.obs['Ground Truth'].isna()]
    scadata = scadata[:,select_genes]
    adata.uns['num_cluster'] = 7
    adata.uns['rad_cutoff'] = 239  
    scadata.obs['celltype'] = meta.loc[scadata.obs_names, 'cluster']
    scadata.obs['cell_type'] = scadata.obs['celltype']

    sc.pp.calculate_qc_metrics(adata, inplace=True)
    scale_factor = adata.obs.total_counts/np.median(adata.obs.total_counts)
    adata.obs['scale_factor'] = scale_factor
    
    cell_type_num = len(scadata.obs['celltype'].unique())
    cell_types = scadata.obs['celltype'].unique()
    word_to_idx_celltype = {word: i for i, word in enumerate(cell_types)}
    idx_to_word_celltype = {i: word for i, word in enumerate(cell_types)}
    celltype_idx = [word_to_idx_celltype[w] for w in scadata.obs['celltype']]
    scadata.obs['cell_type_idx'] = celltype_idx

    pseudo_spot_simulation_paras = {
            'spot_num': adata.shape[0]*15,
            'min_cell_num_in_spot': 5,
            'max_cell_num_in_spot': 8,
            'generation_method': 'celltype',
            'max_cell_types_in_spot': 4,   
        }

    pseudo_adata = pseudo_spot_generation(scadata,
                        idx_to_word_celltype,
                        spot_num = pseudo_spot_simulation_paras['spot_num'],
                        min_cell_number_in_spot = pseudo_spot_simulation_paras['min_cell_num_in_spot'],
                        max_cell_number_in_spot = pseudo_spot_simulation_paras['max_cell_num_in_spot'],
                        max_cell_types_in_spot = pseudo_spot_simulation_paras['max_cell_types_in_spot'],
                        generation_method = pseudo_spot_simulation_paras['generation_method'],
                        n_jobs = 4
                        )

    with open("data/"+data_name, "wb") as f:
        pickle.dump({"sc":scadata,"realST":adata,"pseudoST":pseudo_adata}, f)
    
    return scadata, adata, pseudo_adata