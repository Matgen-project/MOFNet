from copyreg import pickle
import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import time
from featurization.data_utils import load_data_from_df, construct_loader_gf_pressurever, construct_dataset_gf_pressurever, data_prefetcher, load_real_data, construct_dataset_real, load_predict_data
from transformer import make_model
from argparser import parse_predict_args
import logging
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def splitdata(length,fold,index):
    fold_length = length // fold
    index_list = np.arange(length)
    if index == 1:
        val = index_list[:fold_length]
        test = index_list[fold_length * (fold - 1):]
        train = index_list[fold_length : fold_length * (fold - 1)]
    elif index == fold:
        val = index_list[fold_length * (fold - 1):]
        test = index_list[fold_length * (fold - 2) : fold_length * (fold - 1)]
        train = index_list[:fold_length * (fold - 2)]
    else:
        val = index_list[fold_length * (index - 1) : fold_length * index]
        test = index_list[fold_length * (index - 2) : fold_length * (index - 1)]
        train = np.concatenate([index_list[:fold_length * (index - 2)],index_list[fold_length * index:]])
    return train,val,test

def test(model,data_loader,mean,std):
    os.makedirs('data.bak',exist_ok=True)
    model.eval()
    error = 0
    # prefetcher = data_prefetcher(data_loader)
    batch_idx = 0
    # data = prefetcher.next()
    futures, ys = [], []
    # while data is not None:
    #print("data_loader is ")
    #print(data_loader)
    for data in data_loader:
        adjacency_matrix, node_features, distance_matrix, global_features, y = data
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        #adapteral_features.shape[-1] - 9
        adapter_dim = global_features.shape[-1] - 9
        pressure = global_features[...,-adapter_dim:]
        # pressure_itp = (pressure[...,:-1] + pressure[...,1:])/2
        pressure_itp = torch.linspace(pressure.min(), pressure.max(), 30).reshape(1,-1)
        # global_features = torch.cat([global_features,pressure_itp],1)
        # adapter_dim = global_features.shape[-1] - 9
        output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None, global_features, adapter_dim)
        print("output is ")
        print(output)
        gf_itp = torch.cat([global_features[...,:-adapter_dim], pressure_itp], 1)
        adapter_dim_itp = gf_itp.shape[-1] - 9
        output_itp = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None, gf_itp, adapter_dim_itp)
        #y_tmp = y.cpu().detach().numpy().reshape(-1)
        futures_tmp = output.cpu().detach().numpy().reshape(-1)
        #futures_tmp_itp = output_itp.cpu().detach().numpy().reshape(-1) * std + mean
        pres = pressure.cpu().detach().numpy().reshape(-1)
        #pres_itp = pressure_itp.cpu().detach().numpy().reshape(-1)
        # print(pres.shape, pres_itp.shape, y_tmp.shape, futures_tmp.shape, futures_tmp_itp.shape)
        plt.xlabel('log pressure(Pa)')
        plt.ylabel('adsorption(mol/kg)')
#        l1 = plt.scatter(pres, y_tmp, c ='r', marker = 'o')
        l2 = plt.scatter(pres, futures_tmp, c = 'g', marker = 'x')
#        l3 = plt.scatter(pres_itp, futures_tmp_itp, c = 'b', marker = 'x')
        plt.legend(handles=[l2],labels=['label','prediction','interpolation'],loc='best')
        plt.savefig(f'data.bak/{batch_idx + 1}.png')
        plt.cla()
#        ys += list(y_tmp)
        futures += list(futures_tmp)
        batch_idx += 1
        # data = prefetcher.next()

    futures = np.array(futures)
    print(futures)
    return futures
#    ys = np.array(ys)
#    mae = np.mean(np.abs(futures - ys))
#    rmse = np.sqrt(np.mean((futures - ys)**2))
#    pcc = np.corrcoef(futures,ys)[0][1]
#    smape = 2 * np.mean(np.abs(futures-ys)/(np.abs(futures)+np.abs(ys)))
#
#    return {'MAE':mae, 'RMSE':rmse, 'PCC':pcc, 'sMAPE':smape}


def ensemble_test(models,data_loader, mean, std, img_dir, names, p_ori):
    os.makedirs(img_dir,exist_ok=True)
    for model in models:
        model.eval()
    error = 0
    # prefetcher = data_prefetcher(data_loader)
    batch_idx = 0
    # data = prefetcher.next()
    futures, ys = [], []
    p_ori = np.log(float(p_ori))
    # while data is not None:
    for data in data_loader:
        adjacency_matrix, node_features, distance_matrix, global_features, y = data
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        adapter_dim = global_features.shape[-1] - 9
        pressure = global_features[...,-adapter_dim:]
        # pressure_itp = (pressure[...,:-1] + pressure[...,1:])/2
        # pressure_itp = torch.linspace( - pressure.max(), pressure.max() * 2, 30).reshape(1,-1)
        # global_features = torch.cat([global_features,pressure_itp],1)
        # adapter_dim = global_features.shape[-1] - 9
        outputs = []
        for model in models:
            output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None, global_features, adapter_dim)
            outputs.append(output.cpu().detach().numpy().reshape(-1) * std + mean)
        # gf_itp = torch.cat([global_features[...,:-adapter_dim], pressure_itp], 1)
        # adapter_dim_itp = gf_itp.shape[-1] - 9
        # output_itp = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None, gf_itp, adapter_dim_itp)
        y_tmp = y.cpu().detach().numpy().reshape(-1)
        futures_tmp = np.mean(np.array(outputs),axis=0)
        # futures_tmp = output.cpu().detach().numpy().reshape(-1) * std + mean
        # futures_tmp_itp = output_itp.cpu().detach().numpy().reshape(-1) * std + mean
        pres = pressure.cpu().detach().numpy().reshape(-1) + p_ori
        # pres_itp = pressure_itp.cpu().detach().numpy().reshape(-1)
        # print(pres.shape, pres_itp.shape, y_tmp.shape, futures_tmp.shape, futures_tmp_itp.shape)
        
        plt.xlabel('log pressure(Pa)')
        plt.ylabel('adsorption(mol/kg)')
        # l1 = plt.scatter(pres, y_tmp, c ='r', marker = 'o')
        l2 = plt.scatter(pres, futures_tmp, c = 'g', marker = 'x')
        plt.legend(handles=[l2],labels=['prediction'],loc='best')
        # plt.scatter(pres_itp, futures_tmp_itp, c = 'b', marker = 'x')
        plt.savefig(f'{img_dir}/{names[batch_idx]}.png')
        plt.cla()
        ys += list(y_tmp)
        futures += list(futures_tmp)
        batch_idx += 1
        # data = prefetcher.next()

    futures = np.array(futures)
    ys = np.array(ys)
    mae = np.mean(np.abs(futures - ys))
    rmse = np.sqrt(np.mean((futures - ys)**2))
    pcc = np.corrcoef(futures,ys)[0][1]
    smape = 2 * np.mean(np.abs(futures-ys)/(np.abs(futures)+np.abs(ys)))

    return {'MAE':mae, 'RMSE':rmse, 'PCC':pcc, 'sMAPE':smape}

    


def printParams(param_dic, logger=None):
    print("=========== Parameters ==========")
    for k,v in model_params.items():
        print(f'{k} : {v}')
    print("=================================")
    print()
    if logger:
        for k,v in model_params.items():
            logger.info(f'{k} : {v}')

def applyIndexOnList(lis,idx):
    ans = []
    for _ in idx:
        ans.append(lis[_])
    return ans

def set_seed(seed):
    torch.manual_seed(seed)  # set seed for cpu 
    torch.cuda.manual_seed(seed)  # set seed for gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)  # numpy

def get_logger(data_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(data_dir + "/real_test_log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class CheckpointHandler(object):
    def __init__(self, save_dir, max_save=5):
        self.save_dir = save_dir
        self.max_save = max_save
        self.init_info()

    def init_info(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.metric_dic = {}
        if os.path.exists(self.save_dir+'/eval_log.txt'):
            with open(self.save_dir+'/eval_log.txt','r') as f:
                ls = f.readlines()
            for l in ls:
                l = l.strip().split(':')
                assert len(l) == 2
                self.metric_dic[l[0]] = float(l[1])

    
    def save_model(self, model, model_params, epoch, eval_metric):
        max_in_dic = max(self.metric_dic.values()) if len(self.metric_dic) else 1e9
        if eval_metric > max_in_dic:
            return
        if len(self.metric_dic) == self.max_save:
            self.remove_last()
        self.metric_dic['model-'+str(epoch)+'.pt'] = eval_metric
        state = {"params":model_params, "epoch":epoch, "model":model.state_dict()}
        torch.save(state, self.save_dir + '/' + 'model-'+str(epoch)+'.pt')
        log_str = '\n'.join(['{}:{:.7f}'.format(k,v) for k,v in self.metric_dic.items()])
        with open(self.save_dir+'/eval_log.txt','w') as f:
            f.write(log_str)


    def remove_last(self):
        last_model = sorted(list(self.metric_dic.keys()),key = lambda x:self.metric_dic[x])[-1]
        if os.path.exists(self.save_dir+'/'+last_model):
            os.remove(self.save_dir+'/'+last_model)
        self.metric_dic.pop(last_model)

    def checkpoint_best(self):
        best_model = sorted(list(self.metric_dic.keys()),key = lambda x:self.metric_dic[x])[0]
        state = torch.load(self.save_dir + '/' + best_model)
        return state

    def checkpoint_avg(self):
        return_dic = None
        model_num = 0
        tmp_model_params = None
        for ckpt in os.listdir(self.save_dir):
            if not ckpt.endswith('.pt'):
                continue
            model_num += 1
            state = torch.load(self.save_dir + '/' + ckpt)
            model,tmp_model_params = state['model'], state['params']
            if not return_dic:
                return_dic = model
            else:
                for k in return_dic:
                    return_dic[k] += model[k]
        for k in return_dic:
            return_dic[k] = return_dic[k]/model_num
        return {'params':tmp_model_params, 'model':return_dic}

if __name__ == '__main__':

    model_params = parse_predict_args()
    batch_size = 1
    device_ids = [0,1,2,3]

    if model_params['gas_type'] == 'CH4':
        save_dir = f"./mof_adapted_rbf/CH4_5e4/"
    elif model_params['gas_type'] == 'CO2':
        save_dir = f"./mof_adapted_rbf/CO2_1e4/"
    elif model_params['gas_type'] == 'N2':
        save_dir = f'./mof_adapted_rbf/N2_2e2/'

    with open(os.path.join(save_dir,f'offset.p'),'rb') as f:
        p_ori, mean, std, fmean, fstd = pickle.load(f)

    # test_errors_all = []
    # set_seed(model_params['seed'])

    X, f, names = load_predict_data(model_params['data_dir'])
    f = np.array(f)
    f = (f - fmean) / fstd

    data_num = len(names)
    start, end, itp = model_params['pressure']
    y = np.zeros((data_num,itp))
    p = np.linspace(np.log(start), np.log(end), itp)
    p = np.tile(np.exp(p), [data_num, 1])

    models = []
    img_dir = model_params['save_dir']
    for fold_idx in range(1,11):
        save_dir_f = os.path.join(save_dir, f"Fold-{fold_idx}")
        state = CheckpointHandler(save_dir_f).checkpoint_best()
        model = make_model(**state['params'])
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state['model'])
        model = model.to(device)
        models.append(model)
    test_set = construct_dataset_real(X, f, y, p, p_ori)
    test_loader = construct_loader_gf_pressurever(test_set,1,shuffle=False)
    test_res = ensemble_test(models, test_loader, mean, std, img_dir, names, p_ori)
    
    # X, f, y, p, names = load_real_data('data/exp_data', model_params['gas_type'])
    # f = np.array(f)
    # f = (f - fmean) / fstd


    # test_errors = []

    # models = []
    # img_dir = f"images/{model_params['gas_type']}"
    # predict_res = []
    # for fold_idx in range(1,11):
    #     set_seed(model_params['seed'])
    #     if model_params['gas_type'] == 'CH4':
    #         save_dir = f"./mof_adapted_rbf/CH4_5e4/Fold-{fold_idx}"
    #     elif model_params['gas_type'] == 'CO2':
    #         save_dir = f"./mof_adapted_rbf/CO2_1e4/Fold-{fold_idx}"
    #     elif model_params['gas_type'] == 'N2':
    #         save_dir = f'./mof_adapted_rbf/N2_2e2/Fold-{fold_idx}'
    #     state = CheckpointHandler(save_dir).checkpoint_best()
    #     model = make_model(**state['params'])
    #     model = torch.nn.DataParallel(model)
    #     model.load_state_dict(state['model'])
    #     model = model.to(device)
    #     models.append(model)

    # test_set = construct_dataset_real(X, f, y, p, model_params['pressure'])
    # test_loader = construct_loader_gf_pressurever(test_set,1,shuffle=False)
    # test_res = ensemble_test(models, test_loader, mean, std, img_dir, names)
    # predict_res.append(test_res)
    # print("predict_list is ")
    # print(predict_res)
