from cProfile import label
import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import time
from featurization.data_utils import load_data_from_df, construct_loader_gf_pressurever, construct_dataset_gf_pressurever, data_prefetcher, load_real_data, construct_dataset_real
from transformer import make_model
from argparser import parse_train_args
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

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
    ans = {}
    for data in tqdm(data_loader):
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
        l1 = plt.scatter(pres, y_tmp, c ='r', marker = 'o')
        l2 = plt.scatter(pres, futures_tmp, c = 'g', marker = 'x')
        plt.legend(handles=[l1,l2],labels=['label','prediction'],loc='best')
        # plt.scatter(pres_itp, futures_tmp_itp, c = 'b', marker = 'x')
        plt.savefig(f'{img_dir}/{names[batch_idx]}.png')
        plt.cla()
        ans[names[batch_idx]] = {
            'pressure':np.exp(pres),
            'label':y_tmp,
            'pred':futures_tmp
        }
        # ys += list(y_tmp)
        # futures += list(futures_tmp)
        batch_idx += 1
        # data = prefetcher.next()
    return ans
    # futures = np.array(futures)
    # ys = np.array(ys)
    # mae = np.mean(np.abs(futures - ys))
    # rmse = np.sqrt(np.mean((futures - ys)**2))
    # pcc = np.corrcoef(futures,ys)[0][1]
    # smape = 2 * np.mean(np.abs(futures-ys)/(np.abs(futures)+np.abs(ys)))

    # return {'MAE':mae, 'RMSE':rmse, 'PCC':pcc, 'sMAPE':smape}

    


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
        state = torch.load(self.save_dir + '/' + best_model, map_location='cpu')
        return state

    def checkpoint_avg(self):
        return_dic = None
        model_num = 0
        tmp_model_params = None
        for ckpt in os.listdir(self.save_dir):
            if not ckpt.endswith('.pt'):
                continue
            model_num += 1
            state = torch.load(self.save_dir + '/' + ckpt, map_location='cpu')
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

    model_params = parse_train_args()
    batch_size = 1
    device_ids = [0,1,2,3]
    # logger = get_logger(model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}")
    # X, f, y,p = load_data_from_df(model_params['data_dir'],gas_type=model_params['gas_type'], pressure="all",add_dummy_node = True,use_global_features = True)
    # print("X,f,y,p")
    # # print(X,f,y,p)
    # tar_idx = np.where(p==model_params['pressure'])[0][0]
    # print(f'Loaded {len(X)} data.')
    # logger.info(f'Loaded {len(X)} data.')
    # y = np.array(y)
    # mean = y[...,tar_idx].mean()
    # std = y[...,tar_idx].std()
    # f = np.array(f)
    # fmean = f.mean(axis=0)
    # fstd = f.std(axis=0)

    if model_params['gas_type'] == 'CH4':
        save_dir = f"./mof_adapted_rbf/CH4_5e4/"
    elif model_params['gas_type'] == 'CO2':
        save_dir = f"./mof_adapted_rbf/CO2_1e4/"
    elif model_params['gas_type'] == 'N2':
        save_dir = f'./mof_adapted_rbf/N2_2e2/'

    with open(os.path.join(save_dir,f'offset.p'),'rb') as f:
        p_ori, mean, std, fmean, fstd = pickle.load(f)

#    model_params['d_atom'] = X[0][0].shape[1]
#    model_params['d_feature'] = f.shape[-1]
    
#    printParams(model_params,logger)
#    fold_num = model_params['fold']
#    epoch_num = model_params['epoch']
    test_errors_all = []
    set_seed(model_params['seed'])
    
    X, f, y, p, names = load_real_data('data/exp_data', model_params['gas_type'], add=model_params['data_dir'])
    f = np.array(f)
    f = (f - fmean) / fstd


    test_errors = []

    models = []
    # pressure_list = ['5e4','1e5','1.5e5','2e5','5e5','10e5','50e5','100e5']
    img_dir = f"images/{model_params['gas_type']}"
    predict_res = []
    for fold_idx in range(1,11):
        set_seed(model_params['seed'])
        if model_params['gas_type'] == 'CH4':
            save_dir = f"./mof_adapted_rbf/CH4_5e4/Fold-{fold_idx}"
        elif model_params['gas_type'] == 'CO2':
            save_dir = f"./mof_adapted_rbf/CO2_1e4/Fold-{fold_idx}"
        elif model_params['gas_type'] == 'N2':
            save_dir = f'./mof_adapted_rbf/N2_2e2/Fold-{fold_idx}'
        # state = torch.load(model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}/Fold-{fold_idx}.pt")
        state = CheckpointHandler(save_dir).checkpoint_best()
        model = make_model(**state['params'])
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state['model'])
        model = model.to(device)
        models.append(model)

        #train_idx, val_idx, test_idx = splitdata(len(X),fold_num,fold_idx)
        #test_idx = np.arange(1)
    test_set = construct_dataset_real(X, f, y, p, model_params['pressure'])
        # test_set = construct_dataset_gf_pressurever(X, f, y, p, is_train=True, mask_point=model_params['pressure'])
        #print('test_set is :')
        #print(test_set.toStr())
    test_loader = construct_loader_gf_pressurever(test_set,1,shuffle=False)
    # test_res = test(model,test_loader,1,0)[pressure_list.index(model_params['pressure'])]
    test_res = ensemble_test(models, test_loader, mean, std, img_dir, names, model_params['pressure'])
    with open(os.path.join(img_dir,f"res_{model_params['data_dir'].split('.')[0]}.p"),'wb') as f:
        pickle.dump(test_res,f)
    # predict_res.append(test_res)
    # print("predict_list is ")
    # print(predict_res)
    # print("predict result is ")
    # print(np.array(predict_res).mean(axis=0))
        
    # test_error = ensemble_test(models, test_loader, mean, std, f"images/{model_params['gas_type']}", names)
    # for _ in test_error:
    #     print('Fold: {:02d}, Test {}: {:.7f}'.format(fold_idx, _, test_error[_]))
    #     logger.info('Fold: {:02d}, Test {}: {:.7f}'.format(fold_idx, _, test_error[_]))
    # for _ in test_error:
    #     print('Test {}: {:.7f}'.format( _, test_error[_]))
    #     logger.info('Test {}: {:.7f}'.format( _, test_error[_]))
        

    # for _ in test_errors[0]:
    #     ret = [__[_] for __ in test_errors]
    #     print('Test {}: {:.7f}({:.7f})'.format(_, np.mean(ret), np.std(ret)))
    #     logger.info('Test {}: {:.7f}({:.7f})'.format(_, np.mean(ret), np.std(ret)))
