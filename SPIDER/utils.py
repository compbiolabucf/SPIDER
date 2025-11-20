import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import multiprocessing
from tqdm.notebook import tqdm
import random, sklearn.neighbors
import scipy.sparse as sp
from torch_geometric.data import Data

def process_data(adata):
    df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cell2id = dict(zip(cells, range(cells.shape[0])))
    df['Cell1'] = df['Cell1'].map(cell2id)
    df['Cell2'] = df['Cell2'].map(cell2id)

    G = sp.coo_matrix((np.ones(df.shape[0]), (df['Cell1'], df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edge = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edge[0], edge[1]])), x=torch.FloatTensor(adata.X)) 
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edge[0], edge[1]])), x=torch.FloatTensor(adata.X.todense()))
    return data

def generate_a_spot(sc_exp, 
                    min_cell_number_in_spot, 
                    max_cell_number_in_spot,
                    max_cell_types_in_spot,
                    generation_method,
                   ):
    
    if generation_method == 'cell':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_list = list(sc_exp.obs.index.values)
        picked_cells = random.choices(cell_list, k=cell_num)
        return sc_exp[picked_cells]
    elif generation_method == 'celltype':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_type_list = list(sc_exp.obs['cell_type'].unique())
        cell_type_num = random.randint(1, max_cell_types_in_spot)
        
        while(True):
            cell_type_list_selected = random.choices(sc_exp.obs['cell_type'].value_counts().keys(), k=cell_type_num)
            if len(set(cell_type_list_selected)) == cell_type_num:
                break
        sc_exp_filter = sc_exp[sc_exp.obs['cell_type'].isin(cell_type_list_selected)]
        
        picked_cell_type = random.choices(cell_type_list_selected, k=cell_num)
        picked_cells = []
        for i in picked_cell_type:
            data = sc_exp[sc_exp.obs['cell_type'] == i]
            cell_list = list(data.obs.index.values)
            picked_cells.append(random.sample(cell_list, 1)[0])
            
        return sc_exp_filter[picked_cells]
    else:
        print('generation_method should be "cell" or "celltype" ')

        

def pseudo_spot_generation(sc_exp, 
                           idx_to_word_celltype,
                           spot_num, 
                           min_cell_number_in_spot, 
                           max_cell_number_in_spot,
                           max_cell_types_in_spot,
                           generation_method,
                           n_jobs = -1
                          ):
    
    cell_type_num = len(sc_exp.obs['cell_type'].unique())
    
    cores = multiprocessing.cpu_count()
    if n_jobs == -1:
        pool = multiprocessing.Pool(processes=cores)
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
    args = [(sc_exp, min_cell_number_in_spot, max_cell_number_in_spot, max_cell_types_in_spot, generation_method) for i in range(spot_num)]
    generated_spots = pool.starmap(generate_a_spot, tqdm(args, desc='Generating pseudo-spots'))
    
    pseudo_spots = []
    pseudo_spots_table = np.zeros((spot_num, sc_exp.shape[1]), dtype=float)
    pseudo_fraction_table = np.zeros((spot_num, cell_type_num), dtype=float)
    for i in range(spot_num):
        one_spot = generated_spots[i]
        pseudo_spots.append(one_spot)
        pseudo_spots_table[i] = one_spot.X.sum(axis=0)
        for j in one_spot.obs.index:
            type_idx = one_spot.obs.loc[j, 'cell_type_idx']
            pseudo_fraction_table[i, type_idx] += 1
    pseudo_spots_table = pd.DataFrame(pseudo_spots_table, columns=sc_exp.var.index.values)
    pseudo_spots = anndata.AnnData(X=pseudo_spots_table.iloc[:,:].values)
    pseudo_spots.obs.index = pseudo_spots_table.index[:]
    pseudo_spots.var.index = pseudo_spots_table.columns[:]
    type_list = [idx_to_word_celltype[i] for i in range(cell_type_num)]
    pseudo_fraction_table = pd.DataFrame(pseudo_fraction_table, columns=type_list)
    pseudo_fraction_table['cell_num'] = pseudo_fraction_table.sum(axis=1)
    for i in pseudo_fraction_table.columns[:-1]:
        pseudo_fraction_table[i] = pseudo_fraction_table[i]/pseudo_fraction_table['cell_num']
    pseudo_spots.obs = pseudo_spots.obs.join(pseudo_fraction_table)
        
    return pseudo_spots


def Cal_knn_expression(adata, k_cutoff=8, verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    if verbose:
        print('------Calculating Similarity Graph...')

    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(adata.X)
    distances, indices = nbrs.kneighbors(adata.X)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(adata.shape[0]), np.array(adata.obs.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Similarity_Net'] = Spatial_Net

def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='SPIDER', random_seed=2020):
   
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata