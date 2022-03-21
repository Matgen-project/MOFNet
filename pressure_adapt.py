import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import time
from featurization.data_utils import load_data_from_df, construct_loader_gf_pressurever, construct_dataset_gf_pressurever, data_prefetcher
from transformer import make_model
from argparser import parse_finetune_args
import logging

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
        output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None, global_features, adapter_dim)
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
        output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None, global_features, adapter_dim)
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


def printParams(param_dic, logger=None):
    print("=========== Parameters ==========")
    for k,v in model_params.items():
        print(f'{k} : {v}')
    print("=================================")
    print()
    if logger:
        for k,v in model_params.items():
            logger.info(f'{k} : {v}')

def checkpoint_avg(save_dir):
    return_dic = None
    model_num = 0
    tmp_model_params = None
    for ckpt in os.listdir(save_dir):
        if not ckpt.endswith('.pt'):
            continue
        model_num += 1
        state = torch.load(save_dir + '/' + ckpt)
        model,tmp_model_params = state['model'], state['params']
        if not return_dic:
            return_dic = model
        else:
            for k in return_dic:
                return_dic[k] += model[k]
    for k in return_dic:
        return_dic[k] = return_dic[k]/model_num
    return {'params':tmp_model_params, 'model':return_dic}

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

    # model_params['d_atom'] = X[0][0].shape[1]
    # model_params['d_feature'] = f.shape[-1] + 1
    
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

    for fold_idx in range(9, fold_num + 1):

        set_seed(model_params['seed'])
        ori_state = checkpoint_avg(model_params['ori_dir']+f'/Fold-{fold_idx}')
        # ori_state = torch.load(model_params['ori_dir']+f'/Fold-{fold_idx}.pt')
        ori_params = ori_state['params']
        ori_params['adapter_finetune'] = True
        # ori_params['adapter_dim'] = model_params['adapter_dim']
        model = make_model(**ori_params)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.load_state_dict(ori_state['model'],strict=False)
        model = model.to(device)
        lr = model_params['lr']
        # for name, para in model.named_parameters():
        #     if 'generator' not in name:
        #         para.required_grad = False
        # optimizer = torch.optim.Adam([model.module.generator.adapter_vec], lr=lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = get_RdecayFactor(ori_params['warmup_step']))
        best_val_error = 0
        best_val_error_s = 0
        test_error = 0
        best_epoch = -1

        train_idx, val_idx, test_idx = splitdata(len(X),fold_num, fold_idx)

        train_set = construct_dataset_gf_pressurever(applyIndexOnList(X,train_idx), f[train_idx], y[train_idx],p, is_train=True, mask_point=model_params['pressure'])

        # train_loader = construct_loader_gf_pressurever(train_set,batch_size)

        val_set = construct_dataset_gf_pressurever(applyIndexOnList(X,val_idx), f[val_idx], y[val_idx],p, is_train=True, mask_point=model_params['pressure'])

        # val_loader = construct_loader_gf_pressurever(val_set, batch_size, shuffle=False)

        test_set = construct_dataset_gf_pressurever(applyIndexOnList(X,test_idx), f[test_idx], y[test_idx],p, is_train=True, mask_point=model_params['pressure'])
        # test_loader = construct_loader_gf(applyIndexOnList(X,test_idx),f[test_idx], y[test_idx],batch_size)

        ckpt_handler = CheckpointHandler(model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}/Fold-{fold_idx}")
        # ckpt_handler_s = CheckpointHandler(model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}/Fold-{fold_idx}/{model_params['pressure']}")

        for epoch in range(1,epoch_num + 1):
            # if epoch % 4 != 0:
            #     train_set.changeTarPoint(None)
            #     train_adapter_dim = model_params['adapter_dim']
            # else:
            #     train_set.changeTarPoint(model_params['pressure'])
            #     train_adapter_dim = 1
            train_adapter_dim = model_params['adapter_dim']
            train_loader = construct_loader_gf_pressurever(train_set,batch_size)
            loss = train(model, epoch, train_loader,optimizer,scheduler, train_adapter_dim)
            # if pres != model_params['pressure']:
            val_loader = construct_loader_gf_pressurever(val_set, batch_size, shuffle=False)
            val_error = test(model, val_loader, mean, std, train_adapter_dim)['MAE']
            val_error_ = np.mean(val_error)
            # val_error_s = val_error[model_params['pressure']]
            ckpt_handler.save_model(model,ori_params,epoch,val_error_)
            # ckpt_handler_s.save_model(model,ori_params,epoch,val_error_s)

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
                # state = {"params":model_params, "epoch":epoch, "model":model.state_dict()}
                # torch.save(state, model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}/Fold-{fold_idx}.pt")
            # elif best_val_error_s == 0 or val_error_s <= best_val_error_s:
            #     best_val_error_s = val_error_s
            #     for pres in p:
            #         test_set.changeTarPoint(pres)
            #         test_loader = construct_loader_gf_pressurever(test_set, batch_size, shuffle=False)
            #         test_error_s = test(model, test_loader, mean, std)
            #         for _ in test_error_s.keys():
            #             print('Fold: {:02d}, Epoch: {:03d}, Pressure: {}, Test {}: {:.7f}'.format(fold_idx, epoch, pres, _, test_error_s[_]))
            #             logger.info('Fold: {:02d}, Epoch: {:03d}, Pressure: {}, Test {}: {:.7f}'.format(fold_idx, epoch, pres, _, test_error_s[_]))
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