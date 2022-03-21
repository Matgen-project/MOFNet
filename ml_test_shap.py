import numpy as np
from featurization.data_utils import load_data_from_df, construct_loader_gf
from sklearn import tree, svm, ensemble
from featurization.data_utils import load_data_from_df, construct_loader_gf, data_prefetcher
from argparser import parse_train_args,parse_ml_args
import logging
import shap

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


def set_seed(seed):
    np.random.seed(seed)  # numpy

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

def get_logger(save_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(save_dir + "/test_log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_metric_dict(predicted, ground_truth):
    mae = np.mean(np.abs(predicted - ground_truth))
    smape = np.mean(np.abs(predicted - ground_truth) / ((np.abs(ground_truth) + np.abs(predicted)) / 2))
    pcc = np.corrcoef(predicted, ground_truth)[0][1]
    rmse = np.sqrt(np.mean((predicted - ground_truth) ** 2))
    return {'MAE':mae, 'sMAPE':smape, 'PCC': pcc, 'RMSE':rmse}

if __name__ == '__main__':

    model_params = parse_ml_args()
    device_ids = [0,1,2,3]
    logger = get_logger(model_params['save_dir'] + f"/{model_params['ml_type']}/{model_params['gas_type']}_{model_params['pressure']}/")
    X, f, y = load_data_from_df(model_params['data_dir'],gas_type=model_params['gas_type'], pressure=model_params['pressure'],add_dummy_node = True,use_global_features = True)
    print(f'Loaded {len(X)} data.')
    logger.info(f'Loaded {len(X)} data.')
    y = np.array(y).reshape(-1)
    mean = y.mean()
    std = y.std()
    y = (y - mean) / std
    f = np.array(f)
    fmean = f.mean(axis=0)
    fstd = f.std(axis=0)
    f = (f - fmean) / fstd

    # add local
    Xs = [np.mean(_[0][:,1:],axis=0) for _ in X]
    f = np.concatenate((Xs,f),axis=1)
    
    printParams(model_params,logger)
    fold_num = model_params['fold']
    # test_maes, test_rmses,test_pccs = [], [], []
    test_errors = []
    idx_list = np.arange(len(X))
    set_seed(model_params['seed'])
    np.random.shuffle(idx_list)
    X = applyIndexOnList(X,idx_list)
    f = f[idx_list]
    y = y[idx_list]

    for fold_idx in range(1,fold_num + 1):
        set_seed(model_params['seed'])

        train_idx, val_idx, test_idx = splitdata(len(X),fold_num,fold_idx)

        train_f,train_y = f[train_idx], y[train_idx]
        test_f,test_y = f[test_idx], y[test_idx]

        if model_params['ml_type'] == 'RF':

            model = ensemble.RandomForestRegressor(n_estimators=100,criterion='mse',min_samples_split=2,min_samples_leaf=1,max_features='auto')

        elif model_params['ml_type'] == 'SVR':

            model = svm.SVR()

        elif model_params['ml_type'] == 'DT':

            model = tree.DecisionTreeRegressor()

        elif model_params['ml_type'] == 'GBRT':

            model = ensemble.GradientBoostingRegressor()

        model.fit(train_f,train_y)
        # add code for shap analysis
        shap_values = shap.TreeExplainer(model).shap_values(test_f)
        shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(test_f)
        
        shap.summary_plot(shap_values, test_f, plot_type="bar")
        #exit()

        future = model.predict(test_f) * std + mean

        test_y = test_y * std + mean

        # test_mae = np.mean(np.abs(future - test_y))
        # test_rmse = np.sqrt(np.mean((future - test_y)**2))
        # test_pcc = np.corrcoef(future, test_y)[0][1]

        # test_maes.append(test_mae)
        # test_rmses.append(test_rmse)
        # test_pccs.append(test_pcc)
        test_error = get_metric_dict(future, test_y)
        for _ in test_error.keys():
            print('Fold: {:02d}, Test {}: {:.7f}'.format(fold_idx, _, test_error[_]))
            logger.info('Fold: {:02d}, Test {}: {:.7f}'.format(fold_idx, _, test_error[_]))
        test_errors.append(test_error)
        # print('Fold: {:02d}, Test MAE: {:.7f}, Test RMSE: {:.7f}, Test PCC: {:.7f}'.format(fold_idx, test_mae, test_rmse, test_pcc))
        # logger.info('Fold: {:02d}, Test MAE: {:.7f}, Test RMSE: {:.7f}, Test PCC: {:.7f}'.format(fold_idx, test_mae, test_rmse, test_pcc))
    # err_mean = np.mean(test_rmses)
    # err_std  = np.std(test_rmses)
    # output_str = 'Test results of {:02d}-Folds : MAE : {:.7f}({:.7f}), RMSE : {:.7f}({:.7f}), PCC : {:.7f}({:.7f})'.format(fold_num,np.mean(test_maes),np.std(test_maes),np.mean(test_rmses),np.std(test_rmses),np.mean(test_pccs),np.std(test_pccs))
    # print(output_str)
    # logger.info(output_str)
    for _ in test_errors[0].keys():
        err_mean = np.mean([__[_] for __ in test_errors])
        err_std  = np.std([__[_] for __ in test_errors])
        print('Test {} of {:02d}-Folds : {:.7f}({:.7f})'.format(_,fold_num,err_mean,err_std))
        logger.info('Test {} of {:02d}-Folds : {:.7f}({:.7f})'.format(_,fold_num,err_mean,err_std))
