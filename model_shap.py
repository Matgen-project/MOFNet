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
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gradient_shap(model, sample_loader, test_loader, batch_size):
    model.eval()
    model.set_adapter_dim(1)
    graph_reps, global_feas = [],[]
    for data in tqdm(sample_loader):
        adjacency_matrix, node_features, distance_matrix, global_features, y = (_.cpu() for _ in data)
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        batch_mask = batch_mask.float()
        graph_rep = model.encode(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
        graph_reps.append(graph_rep)
        global_feas.append(global_features)
    graph_reps = torch.cat(graph_reps)
    global_feas = torch.cat(global_feas)
    e = shap.GradientExplainer(model.generator, [graph_reps, global_feas])
    shap_all = []
    for data in tqdm(test_loader):
        adjacency_matrix, node_features, distance_matrix, global_features, y = (_.cpu() for _ in data)
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        batch_mask = batch_mask.float()
        graph_rep = model.encode(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
        ans = e.shap_values([graph_rep, global_features],nsamples=10)
        local_shap = np.abs(ans[0].sum(axis=1)).reshape(-1,1)
        global_shap = np.abs(ans[-1])[:,:9]
        shap_values = np.concatenate([local_shap, global_shap],axis=1)
        shap_all.append(shap_values)
    shap_all = np.concatenate(shap_all, axis=0)
    return shap_all

if __name__ == '__main__':
    model_params = parse_train_args()
    device_ids = [0,1,2,3]
    X, f, y, p = load_data_from_df(model_params['data_dir'],gas_type=model_params['gas_type'], pressure='all',add_dummy_node = True,use_global_features = True)
    tar_idx = np.where(p==model_params['pressure'])[0][0]
    print(f'Loaded {len(X)} data.')
    y = np.array(y)
    mean = y[...,tar_idx].mean()
    std = y[...,tar_idx].std()
    y = (y - mean) / std
    f = np.array(f)
    fmean = f.mean(axis=0)
    fstd = f.std(axis=0)
    f = (f - fmean) / fstd
    batch_size = model_params['batch_size']
    fold_num = model_params['fold']
    idx_list = np.arange(len(X))
    set_seed(model_params['seed'])
    np.random.shuffle(idx_list)
    X = applyIndexOnList(X,idx_list)
    f = f[idx_list]
    y = y[idx_list]



    for fold_idx in range(1,2):
        set_seed(model_params['seed'])
        save_dir = model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}/Fold-{fold_idx}"
        ckpt_handler = CheckpointHandler(save_dir)
        state = ckpt_handler.checkpoint_best()
        model = make_model(**state['params'])
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state['model'])
        model = model.module
        train_idx, val_idx, test_idx = splitdata(len(X),fold_num,fold_idx)
        train_sample = construct_dataset_gf_pressurever(applyIndexOnList(X,train_idx), f[train_idx], y[train_idx],p, is_train=False, tar_point=model_params['pressure'],mask_point=model_params['pressure'])
        test_set = construct_dataset_gf_pressurever(applyIndexOnList(X,test_idx), f[test_idx], y[test_idx],p, is_train=False, tar_point=model_params['pressure'],mask_point=model_params['pressure'])
        shaps = {pres:[] for pres in [p[3]]}
        for pres in [p[3]]:
            train_sample.changeTarPoint(pres)
            test_set.changeTarPoint(pres)
            sample_loader = construct_loader_gf_pressurever(train_sample, batch_size, shuffle=False)
            test_loader = construct_loader_gf_pressurever(test_set, batch_size, shuffle=False)
            shap_values = gradient_shap(model, sample_loader, test_loader, batch_size)
            shaps[pres].append(shap_values) 
    
    for pres in [p[3]]:
        shaps[pres] = np.concatenate(shaps[pres],axis=0)

    with open(model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}/shap_result_{p[3]}.p",'wb') as f:
        pickle.dump(shaps, f)

            
