import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import time
from featurization.data_utils import load_data_from_df, construct_loader_gf, data_prefetcher
from transformer import make_model
from argparser import parse_train_args
import logging


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, data_loader, mean, std):
    model.eval()
    error = 0
    prefetcher = data_prefetcher(data_loader)
    batch_idx = 0
    data = prefetcher.next()
    futures, ys = [], []
    while data is not None:
        adjacency_matrix, node_features, distance_matrix, global_features, y = data
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None, global_features)
        ys += list(y.cpu().detach().numpy().reshape(-1))
        futures += list(output.cpu().detach().numpy().reshape(-1))
        batch_idx += 1
        data = prefetcher.next()

    futures = np.array(futures) * std + mean
    ys = np.array(ys) * std + mean
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

def get_logger(save_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(save_dir + "/log.txt")
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

    model_params = parse_train_args()
    batch_size = model_params['batch_size']
    device_ids = [0,1,2,3]
    logger = get_logger(model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}")
    X, f, y = load_data_from_df(model_params['data_dir'],gas_type=model_params['gas_type'], pressure=model_params['pressure'],add_dummy_node = True,use_global_features = True)
    print(f'Loaded {len(X)} data.')
    logger.info(f'Loaded {len(X)} data.')
    y = np.array(y)
    mean = y.mean()
    std = y.std()
    y = (y - mean) / std
    f = np.array(f)
    fmean = f.mean(axis=0)
    fstd = f.std(axis=0)
    f = (f - fmean) / fstd

    model_params['d_atom'] = X[0][0].shape[1]
    model_params['d_feature'] = f.shape[-1]
    
    printParams(model_params,logger)
    fold_num = model_params['fold']
    epoch_num = model_params['epoch']
    test_errors = []
    idx_list = np.arange(len(X))
    set_seed(model_params['seed'])
    np.random.shuffle(idx_list)
    X = applyIndexOnList(X,idx_list)
    f = f[idx_list]
    y = y[idx_list]

    for fold_idx in range(1,fold_num + 1):
        set_seed(model_params['seed'])
        save_dir = model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}/Fold-{fold_idx}"
        # state = torch.load(model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}/Fold-{fold_idx}.pt")
        ckpt_handler = CheckpointHandler(save_dir)
        state = ckpt_handler.checkpoint_best()
        model = make_model(**state['params'])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.load_state_dict(state['model'])
        model = model.to(device)
        train_idx, val_idx, test_idx = splitdata(len(X),fold_num,fold_idx)
        test_loader = construct_loader_gf(applyIndexOnList(X,test_idx),f[test_idx], y[test_idx],batch_size)
        # test_set = construct_dataset_gf_pressurever(applyIndexOnList(X,test_idx), f[test_idx], y[test_idx],p, is_train=False, tar_point=model_params['pressure'],mask_point=model_params['pressure'])
        test_error = test(model, test_loader, mean, std)
        # test_errors = {}
        # for pres in p:
        #     test_set.changeTarPoint(pres)
        #     test_loader = construct_loader_gf_pressurever(test_set, batch_size, shuffle=False)
        #     test_errors[pres] = test(model, test_loader, mean, std)
        #     for _ in test_errors[pres].keys():
        #         print('Fold: {:02d}, Pressure: {}, Test {}: {:.7f}'.format(fold_idx, pres, _, test_errors[pres][_]))
        #         logger.info('Fold: {:02d}, Pressure: {}, Test {}: {:.7f}'.format(fold_idx, pres, _, test_errors[pres][_]))
        # test_errors_all.append(test_errors)
        test_errors.append(test_error)
    for _ in test_errors[0].keys():
        err_mean = np.mean([__[_] for __ in test_errors])
        err_std  = np.std([__[_] for __ in test_errors])
        print('Test {} of {:02d}-Folds : {:.7f}({:.7f})'.format(_,fold_num,err_mean,err_std))
        logger.info('Test {} of {:02d}-Folds : {:.7f}({:.7f})'.format(_,fold_num,err_mean,err_std))