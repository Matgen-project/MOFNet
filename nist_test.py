from cProfile import label
import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import time
from featurization.data_utils import load_data_from_df, construct_loader_gf_pressurever, construct_dataset_gf_pressurever, data_prefetcher, load_real_data, construct_dataset_real
from models.transformer import make_model
from argparser import parse_train_args
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensemble_test(models,data_loader, mean, std, img_dir, names, p_ori):
    os.makedirs(img_dir,exist_ok=True)
    for model in models:
        model.eval()
    batch_idx = 0
    p_ori = np.log(float(p_ori))
    ans = {}
    for data in tqdm(data_loader):
        adjacency_matrix, node_features, distance_matrix, global_features, y = data
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        adapter_dim = global_features.shape[-1] - 9
        pressure = global_features[...,-adapter_dim:]
        outputs = []
        for model in models:
            model.module.set_adapter_dim(adapter_dim)
            output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, global_features)
            outputs.append(output.cpu().detach().numpy().reshape(-1) * std + mean)
        y_tmp = y.cpu().detach().numpy().reshape(-1)
        futures_tmp = np.mean(np.array(outputs),axis=0)
        pres = pressure.cpu().detach().numpy().reshape(-1) + p_ori
        
        plt.xlabel('log pressure(Pa)')
        plt.ylabel('adsorption(mol/kg)')
        l1 = plt.scatter(pres, y_tmp, c ='r', marker = 'o')
        l2 = plt.scatter(pres, futures_tmp, c = 'g', marker = 'x')
        plt.legend(handles=[l1,l2],labels=['label','prediction'],loc='best')
        plt.savefig(f'{img_dir}/{names[batch_idx]}.png')
        plt.cla()
        ans[names[batch_idx]] = {
            'pressure':np.exp(pres),
            'label':y_tmp,
            'pred':futures_tmp
        }
        batch_idx += 1
    return ans

if __name__ == '__main__':

    model_params = parse_train_args()
    batch_size = 1
    device_ids = [0,1,2,3]

    save_dir = f"{model_params['save_dir']}/{model_params['gas_type']}_{model_params['pressure']}"

    with open(os.path.join(save_dir,f'offset.p'),'rb') as f:
        p_ori, mean, std, fmean, fstd = pickle.load(f)

    test_errors_all = []

    X, f, y, p, names = load_real_data(model_params['data_dir'], model_params['gas_type'])
    f = np.array(f)
    f = (f - fmean) / fstd
    test_errors = []
    models = []
    img_dir = os.path.join(model_params['img_dir'],model_params['gas_type'])
    predict_res = []
    for fold_idx in range(1,11):
        save_dir_fold = f"{save_dir}/Fold-{fold_idx}"
        state = CheckpointHandler(save_dir_fold).checkpoint_best()
        model = make_model(**state['params'])
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state['model'])
        model = model.to(device)
        models.append(model)
    test_set = construct_dataset_real(X, f, y, p, p_ori)
    test_loader = construct_loader_gf_pressurever(test_set,1,shuffle=False)
    test_res = ensemble_test(models, test_loader, mean, std, img_dir, names, p_ori)
    with open(os.path.join(img_dir,f"results.p"),'wb') as f:
        pickle.dump(test_res,f)
