
import numpy as np, time, os
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from .utils import process_data
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn,gc
import matplotlib.pyplot as plt
import random
from .model import SPIDER, MMDLoss, ZINB_loss

def set_local_determinism(seed: int = 0, use_cuda: bool = True, strict: bool = True):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(strict, warn_only=not strict)
    except Exception:
        pass

    print(f"[Local deterministic seed set to {seed}]")



def train_SPIDER(adata, psdata, model=None,n_epochs=500, lr=0.00025, key_added='SPIDER',
              gradient_clipping=5.0, weight_decay=0.0001, verbose=True, random_seed=0,
              device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
              W_recon=1.0,W_mmd=1.0,W_cell=1.0,predict=False):
    set_local_determinism(random_seed, use_cuda=True, strict=False) 

    # Set random seeds for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    # Preprocess the data
    adata.X = sp.csr_matrix(adata.X)
    raw_sparse_data = sp.csr_matrix(adata.raw.X)
    raw_data = torch.Tensor(raw_sparse_data.toarray())

    if 'highly_variable' in adata.var.columns:
        adata_vars = adata[:, adata.var['highly_variable']]
    else:
        adata_vars = adata

    if verbose:
        print('Size of Input: ', adata_vars.shape)

    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net does not exist! Run Cal_Spatial_Net first!")

    data = process_data(adata_vars)
    scale_factor = torch.tensor(adata.obs['scale_factor'], dtype=torch.float)
    data.scale_factor = scale_factor.to(device)
    data.raw = raw_data.to(device)

    spot_2_index = {x:i for i,x in enumerate(adata.obs.index)}
    adj_matrix = adata.uns['Spatial_Net'].values[:,:2]
    mapper = np.vectorize(spot_2_index.get)
    edge_index = mapper(adj_matrix).T

    spot_2_index = {x:i for i,x in enumerate(adata.obs.index)}
    adj_matrix = adata.uns['Similarity_Net'].values[:,:2]
    mapper = np.vectorize(spot_2_index.get)
    edge_index_std = mapper(adj_matrix).T

    spot_2_index = {x:i for i,x in enumerate(psdata.obs.index)}
    adj_matrix = psdata.uns['Similarity_Net'].values[:,:2]
    mapper = np.vectorize(spot_2_index.get)
    edge_index_psd = mapper(adj_matrix).T

    edge_index = torch.tensor(edge_index).long().to(device)
    edge_index_psd = torch.tensor(edge_index_psd).to(device)
    edge_index_std = torch.tensor(edge_index_std).to(device)
    
    stx = torch.tensor(adata.X.toarray()).float().to(device)
    psx = torch.tensor(psdata.X).float().to(device)

    y_celltype_st1 = torch.tensor(psdata.obs.values[:,:-1], dtype=torch.float32)
    _, NUM_CELL_TYPES = y_celltype_st1.shape 

    if model is None:    
        model = SPIDER([adata.X.shape[1], 512,30], num_classes = NUM_CELL_TYPES)
    
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    min_loss = float('inf')
    best_model_state = None
    model.train()
    loss_list = []
    recon_loss_list = []
    mmd_loss_list = []
    cell_loss_list = []

    celltype_loss_fn = nn.CrossEntropyLoss()
    mmd_loss_fn = MMDLoss(device=device)
    mse_loss_fn = nn.MSELoss()
    
    if not predict:
        for epoch in tqdm(range(1, n_epochs + 1)):
            
            mean, disp, pi, h2, s2, p2, h2s2, cell_type_ratio  = model(stx, psx, edge_index, edge_index_std, edge_index_psd, data.scale_factor.to(device))
            
            recon_loss = ZINB_loss(data.raw, mean, disp, pi, device=device)

            mmd_loss = mmd_loss_fn(s2,p2)
            cell_loss = celltype_loss_fn(cell_type_ratio,y_celltype_st1.to(device))

            loss = W_recon*recon_loss+W_mmd*mmd_loss+W_cell*cell_loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            loss_list.append(loss.item())
            recon_loss_list.append(recon_loss.item())
            mmd_loss_list.append(mmd_loss.item())
            cell_loss_list.append(cell_loss.item())

            if loss.item() < min_loss:
                min_loss = loss.item()
                best_model_state = model.state_dict().copy()
        model.load_state_dict(best_model_state)

    plt.plot(range(1, n_epochs + 1), loss_list, label='Training Loss')
    plt.plot(range(1, n_epochs + 1), recon_loss_list, label='Recon Loss')
    plt.plot(range(1, n_epochs + 1), mmd_loss_list, label='MMD Loss')
    plt.plot(range(1, n_epochs + 1), cell_loss_list, label='Cell Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    try:
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/Loss_plot.pdf", dpi=300)
    except:
        plt.show()

    model.eval()
    
    mean, disp, pi, h2, s2, p2, h2s2, cell_type_ratio = model(stx, psx, edge_index, edge_index_std, edge_index_psd, data.scale_factor.to(device))
    ReX = mean.cpu().detach().numpy()
    ReX[ReX < 0] = 0
    rep = h2.cpu().detach().numpy()
    rep_comb = h2s2.cpu().detach().numpy()
    adata.obsm[key_added] = rep
    adata.obsm[key_added+"_comb"] = rep_comb
    adata.layers[key_added] = ReX
    adata.uns[key_added + '_loss'] = {"recon_loss":recon_loss_list, "mmd_loss":mmd_loss_list,"cell_loss":cell_loss_list,"total_loss":loss_list}

    return adata, model