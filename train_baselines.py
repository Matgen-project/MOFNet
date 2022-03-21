import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import time
from baselines.data_utils import load_data_from_df, construct_loader, data_prefetcher
from baselines import make_baseline_model
from argparser import parse_baseline_args
import logging

model_params = parse_baseline_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = str(model_params['gpu'])


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


def train(epoch, train_loader, optimizer, scheduler, use_adj=True):
    model.train()
    loss = 0
    loss_all = 0
    prefetcher = data_prefetcher(train_loader, device)
    batch_idx = 0
    data = prefetcher.next()
    while data is not None:
        lr = scheduler.optimizer.param_groups[0]['lr']
        if use_adj:
            node_features, pos, adj, global_feature, y = data
        else:
            node_features, pos, nbh, nbh_mask, global_feature, y = data
            adj = (nbh, nbh_mask)
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

        optimizer.zero_grad()
        output = model(node_features, batch_mask, pos, adj, global_feature)
        y = y.squeeze(-1)
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


def test(data_loader, mean, std, use_adj=True):
    model.eval()
    error = 0
    prefetcher = data_prefetcher(data_loader, device)
    batch_idx = 0
    data = prefetcher.next()
    futures, ys = [], []
    while data is not None:
        
        if use_adj:
            node_features, pos, adj, global_feature, y = data
        else:
            node_features, pos, nbh, nbh_mask, global_feature, y = data
            adj = (nbh, nbh_mask)
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

        optimizer.zero_grad()
        output = model(node_features, batch_mask, pos, adj, global_feature)
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

    # model_params = parse_baseline_args()
    model_name = model_params['model_name']
    if model_name == 'egnn' or 'dimenetpp':
        use_adj = True
    else:
        use_adj = False
    batch_size = model_params['batch_size']
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(model_params['gpu'])
    # device_ids = [0]
    device_ids = [0,1,2,3]
    logger = get_logger(model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}")
    X, f, y = load_data_from_df(model_params['data_dir'],gas_type=model_params['gas_type'], pressure=model_params['pressure'],use_global_features = True)
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
        model = make_baseline_model(**model_params)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)
        lr = model_params['lr']
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = warmupRdecayFactor)
        best_val_error = 0
        test_error = 0
        best_epoch = -1
        train_idx, val_idx, test_idx = splitdata(len(X),fold_num,fold_idx)

        train_loader = construct_loader(applyIndexOnList(X,train_idx), f[train_idx], y[train_idx],batch_size, shuffle=True, use_adj=use_adj)
        val_loader =  construct_loader(applyIndexOnList(X,val_idx), f[val_idx], y[val_idx],batch_size, shuffle=False, use_adj=use_adj)
        test_loader = construct_loader(applyIndexOnList(X,test_idx),f[test_idx], y[test_idx],batch_size, shuffle=False, use_adj=use_adj)

        ckpt_handler = CheckpointHandler(model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}/Fold-{fold_idx}")

        for epoch in range(1,epoch_num + 1):
            loss = train(epoch,train_loader,optimizer,scheduler, use_adj=use_adj)
            val_error = test(val_loader, mean, std, use_adj=use_adj)['MAE']
            ckpt_handler.save_model(model,model_params,epoch,val_error)
            if best_val_error == 0 or val_error <= best_val_error:
                print("Enter test step.\n")
                best_epoch = epoch
                test_error = test(test_loader, mean, std, use_adj=use_adj)
                best_val_error = val_error
                state = {"params":model_params, "epoch":epoch, "model":model.state_dict()}
            #     torch.save(state, model_params['save_dir'] + f"/{model_params['gas_type']}_{model_params['pressure']}/Fold-{fold_idx}.pt")
            lr = scheduler.optimizer.param_groups[0]['lr']

            epoch_op_str = 'Fold: {:02d}, Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, \
                Test MAE: {:.7f}, Test RMSE: {:.7f}, Test PCC: {:.7f}, Test sMAPE: {:.7f}, Best Val MAE {:.7f}(epoch {:03d})'.format(fold_idx, epoch, lr, loss, val_error, test_error['MAE'], test_error['RMSE'], test_error['PCC'], test_error['sMAPE'], best_val_error, best_epoch)

            print(epoch_op_str)

            logger.info(epoch_op_str)
        
        test_errors.append(test_error)
        print('Fold: {:02d}, Test MAE: {:.7f}, Test RMSE: {:.7f}, Test PCC: {:.7f}, Test sMAPE: {:.7f}'.format(fold_idx, test_error['MAE'], test_error['RMSE'], test_error['PCC'], test_error['sMAPE']))
        logger.info('Fold: {:02d}, Test MAE: {:.7f}, Test RMSE: {:.7f}, Test PCC: {:.7f}, Test sMAPE: {:.7f}'.format(fold_idx, test_error['MAE'], test_error['RMSE'], test_error['PCC'], test_error['sMAPE']))
    for _ in test_errors[0].keys():
        err_mean = np.mean([__[_] for __ in test_errors])
        err_std  = np.std([__[_] for __ in test_errors])
        print('Test {} of {:02d}-Folds : {:.7f}({:.7f})'.format(_,fold_num,err_mean,err_std))
        logger.info('Test {} of {:02d}-Folds : {:.7f}({:.7f})'.format(_,fold_num,err_mean,err_std))
