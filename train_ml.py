import numpy as np
from sklearn import tree, svm, ensemble
from featurization.data_utils import load_data_from_df, construct_loader_gf, data_prefetcher
from argparser import parse_train_args,parse_ml_args
from utils import *

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

    Xs = [np.mean(_[0][:,1:],axis=0) for _ in X]
    f = np.concatenate((Xs,f),axis=1)
    
    printParams(model_params,logger)
    fold_num = model_params['fold']
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

        future = model.predict(test_f) * std + mean

        test_y = test_y * std + mean
        test_error = get_metric_dict(future, test_y)
        for _ in test_error.keys():
            print('Fold: {:02d}, Test {}: {:.7f}'.format(fold_idx, _, test_error[_]))
            logger.info('Fold: {:02d}, Test {}: {:.7f}'.format(fold_idx, _, test_error[_]))
        test_errors.append(test_error)
    for _ in test_errors[0].keys():
        err_mean = np.mean([__[_] for __ in test_errors])
        err_std  = np.std([__[_] for __ in test_errors])
        print('Test {} of {:02d}-Folds : {:.7f}({:.7f})'.format(_,fold_num,err_mean,err_std))
        logger.info('Test {} of {:02d}-Folds : {:.7f}({:.7f})'.format(_,fold_num,err_mean,err_std))