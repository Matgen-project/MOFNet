import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset, dataset
from scipy.sparse import coo_matrix
import json
import copy


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
IntTensor = torch.IntTensor
DoubleTensor = torch.DoubleTensor

def load_data_from_df(dataset_path, gas_type, pressure, use_global_features=False):
    print(dataset_path + f'/label/{gas_type}/{gas_type}_ads_all.csv')
    data_df = pd.read_csv(dataset_path + f'/label/{gas_type}/{gas_type}_ads_all.csv',header=0)
    
    data_x = data_df['name'].values
    if pressure == 'all':
        data_y = data_df.iloc[:,1:].values
    else:
        data_y = data_df[pressure].values

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)

    x_all, y_all, name_all = load_data_from_processed(dataset_path, data_x, data_y)

    if use_global_features:
        f_all = load_data_with_global_features(dataset_path, name_all, gas_type)
        if pressure == 'all':
            return x_all, f_all, y_all, data_df.columns.values[1:]
        return x_all, f_all, y_all

    if pressure == 'all':
        return x_all, y_all, data_df.columns.values[1:]
    return x_all, y_all

def load_data_with_global_features(dataset_path, processed_files, gas_type):
    global_feature_path = dataset_path + f'/label/{gas_type}/{gas_type}_global_features_update.csv'
    data_df = pd.read_csv(global_feature_path,header=0)
    data_x = data_df.iloc[:, 0].values
    data_f = data_df.iloc[:,1:].values.astype(np.float32)
    data_dict = {}
    for i in range(data_x.shape[0]):
        data_dict[data_x[i]] = data_f[i]
    f_all = [data_dict[_] for _ in processed_files]
    return f_all



def load_data_from_processed(dataset_path, processed_files, labels):
    x_all, y_all, name_all = [], [], []

    for files, label in zip(processed_files, labels):
        
        data_file = dataset_path + '/processed_en/' + files + '.p'
        try:
            afm, row, col, pos = pickle.load(open(data_file, "rb"))
            x_all.append([afm, row, col, pos])
            y_all.append([label])
            name_all.append(files)
        except:
            pass

    return x_all, y_all, name_all

class MOF:

    def __init__(self, x, y, index, feature = None):
        self.node_features = x[0]
        self.edges = np.array([x[1],x[2]])
        self.pos = x[3]
        self.y = y
        self.index = index
        self.global_feature = feature
        self.size = x[0].shape[0]
        self.adj, self.nbh, self.nbh_mask = self.neighbor_matrix()

    def neighbor_matrix(self):
        csr = coo_matrix((np.ones_like(self.edges[0]), self.edges), shape=(self.size, self.size)).tocsr()
        rowptr, col = csr.indptr, csr.indices
        degree = rowptr[1:] - rowptr[:-1]
        max_d = degree.max()
        _range = np.tile(np.arange(max_d),(self.size,1)).reshape(-1)
        _degree = degree.repeat(max_d).reshape(-1)
        mask = _range < _degree
        ret_nbh = np.zeros(self.size * max_d)
        ret_nbh[mask] = col
        return csr.toarray(), ret_nbh.reshape(self.size, max_d), mask.reshape(self.size, max_d)


class MOFDataset(Dataset):

    def __init__(self, data_list):
        
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MOFDataset(self.data_list[key])
        return self.data_list[key]

def construct_dataset_gf(x_all, f_all, y_all):
    output = [MOF(data[0], data[2], i, data[1])
              for i, data in enumerate(zip(x_all, f_all, y_all))]
    return MOFDataset(output)

def pad_array(array, shape, dtype=np.float32):
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array

def mof_collate_func_adj(batch):
    pos_list, features_list,global_features_list = [], [], []
    adjs = []
    labels = []

    max_size = 0
    for molecule in batch:
        if type(molecule.y[0]) == np.ndarray:
            labels.append(molecule.y[0])
        else:
            labels.append(molecule.y)
        if molecule.node_features.shape[0] > max_size:
            max_size = molecule.node_features.shape[0]

    for molecule in batch:
        pos_list.append(pad_array(molecule.pos, (max_size, 3)))
        features_list.append(pad_array(molecule.node_features, (max_size, molecule.node_features.shape[1])))
        adjs.append(pad_array(molecule.adj, (max_size, max_size)))
        global_features_list.append(molecule.global_feature)

    return [FloatTensor(features_list), FloatTensor(pos_list), LongTensor(adjs), FloatTensor(global_features_list), FloatTensor(labels)]  

def mof_collate_func_nbh(batch):
    pos_list, features_list, global_features_list = [], [], []
    nbhs, nbh_masks = [],[]
    labels = []

    max_size = 0
    max_degree = 0
    for molecule in batch:
        if type(molecule.y[0]) == np.ndarray:
            labels.append(molecule.y[0])
        else:
            labels.append(molecule.y)
        if molecule.node_features.shape[0] > max_size:
            max_size = molecule.node_features.shape[0]
        if molecule.nbh.shape[1] > max_degree:
            max_degree = molecule.nbh.shape[1]

    for molecule in batch:
        pos_list.append(pad_array(molecule.pos, (max_size, 3)))
        features_list.append(pad_array(molecule.node_features, (max_size, molecule.node_features.shape[1])))
        nbhs.append(pad_array(molecule.nbh, (max_size, max_degree)))
        nbh_masks.append(pad_array(molecule.nbh_mask, (max_size, max_degree)))
        global_features_list.append(molecule.global_feature)

    return [FloatTensor(features_list), FloatTensor(pos_list), LongTensor(nbhs), FloatTensor(nbh_masks), FloatTensor(global_features_list), FloatTensor(labels)]  

def construct_loader(x, f, y, batch_size, shuffle=True, use_adj=True):
    data_set = construct_dataset_gf(x, f, y)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                         batch_size=batch_size,
					                     num_workers=8,
                                         collate_fn=mof_collate_func_adj if use_adj else mof_collate_func_nbh,
					                     pin_memory=True,
                                         shuffle=shuffle)
    return loader

class data_prefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(device)
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data = tuple(_.cuda(non_blocking=True) for _ in self.next_data)
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_data
        self.preload()
        return batch