import shap 
import torch
from collections import defaultdict
from featurization.data_utils import load_data_from_df, construct_loader_gf_pressurever, construct_dataset_gf_pressurever, data_prefetcher
from models.transformer import make_model
import numpy as np
import os
from argparser import parse_train_args
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

periodic_table = ('Dummy','H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
                  'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Te', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 
                  'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 
                  'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Unk')


model_params = parse_train_args()
img_dir = os.path.join(model_params['img_dir'],'attn')
os.makedirs(img_dir,exist_ok=True)


def heapmap(atoms, attn, name):
    plt.cla()
    f, ax = plt.subplots(figsize=(20, 15))
    colormap = 'Reds'
    h = sns.heatmap(attn, vmax=attn.max(), yticklabels = atoms, xticklabels = atoms, square=True, cmap=colormap, cbar=False)
    fontsize = 15
    cb=h.figure.colorbar(h.collections[0]) 
    cb.ax.tick_params(labelsize=fontsize) 
    ax.tick_params(labelsize=fontsize,rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(os.path.join(img_dir, name + '.pdf'))

def test(model, data_loader, name_list):
    model.eval()
    batch_idx = -1
    ans = {}
    for data in tqdm(data_loader):
        batch_idx += 1
        adjacency_matrix, node_features, distance_matrix, global_features, y = (_.cpu() for _ in data)
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        graph_rep = model.encode(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
        attn = model.encoder.layers[0].self_attn.self_attn.detach().cpu().numpy()
        atoms = node_features.numpy()[:,:,:83].argmax(axis=-1).reshape(-1)
        attn = attn[0].mean(axis=0)
        atoms = applyIndexOnList(periodic_table, atoms)
        ans[name_list[batch_idx]] = {
            'atoms':atoms,
            'attn':attn
        }
        heapmap(atoms, attn, name_list[batch_idx])
    return ans

if __name__ == '__main__':
    batch_size = 1
    device_ids = [0,1,2,3]
    X, f, y,p = load_data_from_df(model_params['data_dir'],gas_type=model_params['gas_type'], pressure="all",add_dummy_node = True,use_global_features = True, return_names=True)
    print("X,f,y,p")
    tar_idx = np.where(p==model_params['pressure'])[0][0]
    y = np.array(y)
    mean = y[...,tar_idx].mean()
    std = y[...,tar_idx].std()
    f = np.array(f)
    fmean = f.mean(axis=0)
    fstd = f.std(axis=0)
    test_errors_all = []
    f = (f - fmean) / fstd
    X, names = X

    print(f'Loaded {len(X)} data.')

    fold_idx = 1
    save_dir = model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}/Fold-{fold_idx}"
    ckpt_handler = CheckpointHandler(save_dir)
    state = ckpt_handler.checkpoint_best(use_cuda=False)
    model = make_model(**state['params'])
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state['model'])
    model = model.module
    if model_params['name'] == '':
        sample_idx = np.arange(1000)
        tar_name = 'all'
    else:
        if model_params['name'] in names:
            sample_idx = [names.index(model_params['name'])]
            tar_name = model_params['name']
        else:
            sample_idx = [0]
            tar_name = 'random'
    train_sample = construct_dataset_gf_pressurever(applyIndexOnList(X,sample_idx), f[sample_idx], y[sample_idx],p, is_train=False, tar_point=model_params['pressure'],mask_point=model_params['pressure'])
    sample_loader = construct_loader_gf_pressurever(train_sample, 1, shuffle=False)
    ans = test(model, sample_loader, applyIndexOnList(names, sample_idx))

    with open(os.path.join(img_dir,f"attn_{tar_name}.p"),'wb') as f:
        pickle.dump(ans, f)

