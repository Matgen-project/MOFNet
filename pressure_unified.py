import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import time
from featurization.data_utils import load_data_from_df, construct_loader_gf_pressurever, construct_dataset_gf_pressurever, data_prefetcher
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


def warmupRdecayFactor(step):
    warmup_step = model_params['warmup_step']
    if step < warmup_step:
        return step / warmup_step
    else:
        return (warmup_step / step) ** 0.5

def warmupLdecayFactor(step):
    warmup_step = model_params['warmup_step']
    if step < warmup_step:
        return step / warmup_step
    else:
        return warmup_step / step


def train(model, epoch, train_loader, optimizer, scheduler):
    model.train()
    loss = 0
    loss_all = 0
    prefetcher = data_prefetcher(train_loader)
    batch_idx = 0
    data = prefetcher.next()
    while data is not None:
        lr = scheduler.optimizer.param_groups[0]['lr']
        adjacency_matrix, node_features, distance_matrix, global_features, y = data
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

        optimizer.zero_grad()
        output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None, global_features)
        loss = F.mse_loss(output, y)
        loss.backward()
        step_loss = loss.cpu().detach().numpy()
        loss_all += step_loss
        optimizer.step()
        scheduler.step()
        print(f'After Step {batch_idx} of Epoch {epoch}, Loss = {step_loss}, Lr = {lr}')
        batch_idx += 1
        data = prefetcher.next()
    return loss_all / len(train_loader.dataset)



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

if __name__ == '__main__':

    model_params = parse_train_args()
    batch_size = model_params['batch_size']
    device_ids = [0,1,2,3]
    logger = get_logger(model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}")
    X, f, y, p = load_data_from_df(model_params['data_dir'],gas_type=model_params['gas_type'], pressure='all',add_dummy_node = True,use_global_features = True)
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
    model_params['d_feature'] = f.shape[-1] + 1
    
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
    test_errors = []

    for fold_idx in range(1, fold_num + 1):

        set_seed(model_params['seed'])
        model = make_model(**model_params)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)
        lr = model_params['lr']
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = warmupRdecayFactor)
        best_val_error = 0
        test_error = 0
        best_epoch = -1

        train_idx, val_idx, test_idx = splitdata(len(X),fold_num, fold_idx)

        train_set = construct_dataset_gf_pressurever(applyIndexOnList(X,train_idx), f[train_idx], y[train_idx],p, is_train=True, mask_point=model_params['pressure'])

        train_loader = construct_loader_gf_pressurever(train_set,batch_size)

        val_set = construct_dataset_gf_pressurever(applyIndexOnList(X,val_idx), f[val_idx], y[val_idx],p, is_train=False, tar_point=model_params['pressure'])

        # val_loader = construct_loader_gf_pressurever(val_set, batch_size, shuffle=False)

        test_set = construct_dataset_gf_pressurever(applyIndexOnList(X,test_idx), f[test_idx], y[test_idx],p, is_train=False, tar_point=model_params['pressure'])
        # test_loader = construct_loader_gf(applyIndexOnList(X,test_idx),f[test_idx], y[test_idx],batch_size)

        test_error = {}
        val_error = {}
        for epoch in range(1,epoch_num + 1):
            loss = train(model, epoch, train_loader,optimizer,scheduler)
            for pres in p:
                if pres != model_params['pressure']:
                    val_set.changeTarPoint(pres)
                    val_loader = construct_loader_gf_pressurever(val_set, batch_size, shuffle=False)
                    val_error[pres] = test(model, val_loader, mean, std)['MAE']
            val_error_ = np.mean(list(val_error.values()))
            if best_val_error == 0 or val_error_ <= best_val_error:
                print("Enter test step.\n")
                best_epoch = epoch
                best_val_error = val_error_
                for pres in p:
                    test_set.changeTarPoint(pres)
                    test_loader = construct_loader_gf_pressurever(test_set, batch_size, shuffle=False)
                    test_error[pres] = test(model, test_loader, mean, std)
                    for _ in test_error[pres].keys():
                        print('Fold: {:02d}, Epoch: {:03d}, Pressure: {}, Test {}: {:.7f}'.format(fold_idx, epoch, pres, _, test_error[pres][_]))
                        logger.info('Fold: {:02d}, Epoch: {:03d}, Pressure: {}, Test {}: {:.7f}'.format(fold_idx, epoch, pres, _, test_error[pres][_]))
                state = {"params":model_params, "epoch":epoch, "model":model.state_dict()}
                torch.save(state, model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}/Fold-{fold_idx}.pt")
            lr = scheduler.optimizer.param_groups[0]['lr']
            p_str = 'Fold: {:02d}, Epoch: {:03d}, Val MAE: {:.7f}, Best Val MAE: {:.7f}'.format(fold_idx, epoch, val_error_, best_val_error)
            print(p_str)
            logger.info(p_str)

        for pres in test_error:
            for _ in test_error[pres].keys():
                print('Fold: {:02d}, Epoch: {:03d}, Pressure: {}, Test {}: {:.7f}'.format(fold_idx, epoch, pres, _, test_error[pres][_]))
                logger.info('Fold: {:02d}, Epoch: {:03d}, Pressure: {}, Test {}: {:.7f}'.format(fold_idx, epoch, pres, _, test_error[pres][_]))   

        test_errors.append(test_error)  

    for pres in test_errors[0]:
        for _ in test_errors[0][pres].keys():
            mt_list = [__[pres][_] for __ in test_errors]
            p_str = 'Pressure {}, Test {} of {:02d}-Folds: {:.7f}({:.7f})'.format(pres, _, fold_num, np.mean(mt_list), np.std(mt_list))
            print(p_str)
            logger.info(p_str)