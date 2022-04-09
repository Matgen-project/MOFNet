import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import time
from featurization.data_utils import load_data_from_df, construct_loader_gf_pressurever, construct_dataset_gf_pressurever, data_prefetcher
from models.transformer import make_model
from argparser import parse_finetune_args
import pickle
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, epoch, train_loader, optimizer, scheduler, adapter_dim):
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
        output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, global_features)
        loss = F.mse_loss(output.reshape(-1), y.reshape(-1))
        loss.backward()
        step_loss = loss.cpu().detach().numpy()
        loss_all += step_loss
        optimizer.step()
        scheduler.step()
        print(f'After Step {batch_idx} of Epoch {epoch}, Loss = {step_loss}, Lr = {lr}')
        batch_idx += 1
        data = prefetcher.next()
    return loss_all / len(train_loader.dataset)



def test(model, data_loader, mean, std, adapter_dim):
    model.eval()
    error = 0
    prefetcher = data_prefetcher(data_loader)
    batch_idx = 0
    data = prefetcher.next()
    futures, ys = None, None 
    while data is not None:
        adjacency_matrix, node_features, distance_matrix, global_features, y = data
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, global_features)
        output = output.reshape(y.shape).cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        ys = y if ys is None else np.concatenate([ys,y], axis=0)
        futures = output if futures is None else np.concatenate([futures,output], axis=0)
        batch_idx += 1
        data = prefetcher.next()

    futures = np.array(futures) * std + mean
    ys = np.array(ys) * std + mean
    mae = np.mean(np.abs(futures - ys), axis=0)
    rmse = np.sqrt(np.mean((futures - ys)**2, axis=0))
    # pcc = np.corrcoef(futures,ys)[0][1]
    pcc = np.array([np.corrcoef(futures[:,i],ys[:,i])[0][1] for i in range(adapter_dim)])
    smape = 2 * np.mean(np.abs(futures-ys)/(np.abs(futures)+np.abs(ys)), axis=0)

    return {'MAE':mae, 'RMSE':rmse, 'PCC':pcc, 'sMAPE':smape}



def get_RdecayFactor(warmup_step):

    def warmupRdecayFactor(step):
        if step < warmup_step:
            return step / warmup_step
        else:
            return (warmup_step / step) ** 0.5

    return warmupRdecayFactor

if __name__ == '__main__':

    model_params = parse_finetune_args()
    batch_size = model_params['batch_size']
    device_ids = [0,1,2,3]
    logger = get_logger(model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}")
    X, f, y, p = load_data_from_df(model_params['data_dir'],gas_type=model_params['gas_type'], pressure='all',add_dummy_node = True,use_global_features = True)
    tar_idx = np.where(p==model_params['pressure'])[0][0]
    print(f'Loaded {len(X)} data.')
    logger.info(f'Loaded {len(X)} data.')
    y = np.array(y)
    mean = y[...,tar_idx].mean()
    std = y[...,tar_idx].std()
    y = (y - mean) / std
    f = np.array(f)
    fmean = f.mean(axis=0)
    fstd = f.std(axis=0)
    f = (f - fmean) / fstd

    with open(os.path.join(model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}",f'offset.p'),'wb') as file:
        pickle.dump((model_params['pressure'], mean, std, fmean, fstd), file)
    
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
        ori_state = CheckpointHandler(model_params['ori_dir']+f'/Fold-{fold_idx}').checkpoint_avg()
        ori_params = ori_state['params']
        ori_params['adapter_finetune'] = True
        model = make_model(**ori_params)
        model.set_adapter_dim(model_params['adapter_dim'])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.load_state_dict(ori_state['model'],strict=False)
        model = model.to(device)
        lr = model_params['lr']
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = get_RdecayFactor(ori_params['warmup_step']))
        best_val_error = 0
        best_val_error_s = 0
        test_error = 0
        best_epoch = -1

        train_idx, val_idx, test_idx = splitdata(len(X),fold_num, fold_idx)

        train_set = construct_dataset_gf_pressurever(applyIndexOnList(X,train_idx), f[train_idx], y[train_idx],p, is_train=True, mask_point=model_params['pressure'])


        val_set = construct_dataset_gf_pressurever(applyIndexOnList(X,val_idx), f[val_idx], y[val_idx],p, is_train=True, mask_point=model_params['pressure'])


        test_set = construct_dataset_gf_pressurever(applyIndexOnList(X,test_idx), f[test_idx], y[test_idx],p, is_train=True, mask_point=model_params['pressure'])

        ckpt_handler = CheckpointHandler(model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}/Fold-{fold_idx}")

        for epoch in range(1,epoch_num + 1):
            train_adapter_dim = model_params['adapter_dim']
            train_loader = construct_loader_gf_pressurever(train_set,batch_size)
            loss = train(model, epoch, train_loader,optimizer,scheduler, train_adapter_dim)
            val_loader = construct_loader_gf_pressurever(val_set, batch_size, shuffle=False)
            val_error = test(model, val_loader, mean, std, train_adapter_dim)['MAE']
            val_error_ = np.mean(val_error)
            ckpt_handler.save_model(model,ori_params,epoch,val_error_)

            if best_val_error == 0 or val_error_ <= best_val_error:
                print("Enter test step.\n")
                best_epoch = epoch
                best_val_error = val_error_
                test_loader = construct_loader_gf_pressurever(test_set, batch_size, shuffle=False)
                test_error = test(model, test_loader, mean, std, train_adapter_dim)
                for idx, pres in enumerate(p):
                    for _ in test_error.keys():
                        print('Fold: {:02d}, Epoch: {:03d}, Pressure: {}, Test {}: {:.7f}'.format(fold_idx, epoch, pres, _, test_error[_][idx]))
                        logger.info('Fold: {:02d}, Epoch: {:03d}, Pressure: {}, Test {}: {:.7f}'.format(fold_idx, epoch, pres, _, test_error[_][idx]))
            lr = scheduler.optimizer.param_groups[0]['lr']
            p_str = 'Fold: {:02d}, Epoch: {:03d}, Val MAE: {:.7f}, Best Val MAE: {:.7f}'.format(fold_idx, epoch, val_error_, best_val_error)
            print(p_str)
            logger.info(p_str)

        for idx, pres in enumerate(p):
            for _ in test_error.keys():
                print('Fold: {:02d}, Epoch: {:03d}, Pressure: {}, Test {}: {:.7f}'.format(fold_idx, epoch, pres, _, test_error[_][idx]))
                logger.info('Fold: {:02d}, Epoch: {:03d}, Pressure: {}, Test {}: {:.7f}'.format(fold_idx, epoch, pres, _, test_error[_][idx])) 

        test_errors.append(test_error)  

    for idx, pres in enumerate(p):
        for _ in test_errors[0].keys():
            mt_list = [__[_][idx] for __ in test_errors]
            p_str = 'Pressure {}, Test {} of {:02d}-Folds: {:.7f}({:.7f})'.format(pres, _, fold_num, np.mean(mt_list), np.std(mt_list))
            print(p_str)
            logger.info(p_str)