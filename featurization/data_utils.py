import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, dataset
import json
import copy


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
IntTensor = torch.IntTensor
DoubleTensor = torch.DoubleTensor

def cutname(ori_name):
    ori_name = ori_name[:-3]
    if ori_name.endswith('out.'):
        ori_name = ori_name[:-4]
    elif ori_name.endswith('faps.'):
        ori_name = ori_name[:-5]
    return ori_name + 'p'


def load_data_from_df(dataset_path, gas_type, pressure, add_dummy_node=True, use_global_features=False, return_names=False):

    data_df = pd.read_csv(dataset_path + f'/label_by_GCMC/{gas_type}_ads_all.csv',header=0)
    data_x = data_df['name'].values
    if pressure == 'all':
        data_y = data_df.iloc[:,1:].values
    else:
        data_y = data_df[pressure].values

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)

    x_all, y_all, name_all = load_data_from_processed(dataset_path, data_x, data_y, add_dummy_node=add_dummy_node)

    if return_names:
        x_all = (x_all, name_all)

    if use_global_features:
        f_all = load_data_with_global_features(dataset_path, name_all, gas_type)
        if pressure == 'all':
            return x_all, f_all, y_all, data_df.columns.values[1:]
        return x_all, f_all, y_all

    if pressure == 'all':
        return x_all, y_all, data_df.columns.values[1:]
    return x_all, y_all

def norm_str(ori):
    ori = ori.split('.')[0].split('-')
    if ori[-1] == 'clean':
        ori = ori[:-1]
    elif ori[-2] == 'clean':
        ori = ori[:-2]
    return '-'.join(ori[1:])


def load_real_data(dataset_path, gas_type):
    
    data_df = pd.read_csv(dataset_path + f'/global_features/exp_geo_all.csv', header=0)
    data_x = data_df['name'].values
    data_y = data_df.iloc[:,1:].values
    global_dic = {}
    for x,y in zip(data_x, data_y):
        global_dic[x] = y
    with open(dataset_path + '/isotherm_data/all.json') as f:
        labels = json.load(f)[gas_type]['data']
    label_dict = {_['name']:_["isotherm_data"] for _ in labels}
    
    with open(dataset_path + f'/isotherm_data/{gas_type}.txt','r') as f:
        ls = f.readlines()
    ls = [_.strip().split() for _ in ls]
    X_all, y_all, f_all, p_all, n_all = [],[],[],[],[]
    for l in ls:
        if l[0] not in global_dic:
            continue
        gf = global_dic[l[0]]
        afm, adj, dist = pickle.load(open(dataset_path + f'/local_features/{l[0]}.cif.p', "rb"))
        afm, adj, dist = add_dummy_node_func(afm, adj, dist)
        iso = label_dict[norm_str(l[0])]
        p,y = [],[]
        for _ in iso:
            if _['pressure'] > 0:
                p.append(_['pressure'])
                y.append(_['adsorption'])
        if len(p) == 0:
            continue
        X_all.append([afm,adj,dist])
        f_all.append(gf)
        p_all.append(p)
        y_all.append(y)
        n_all.append(norm_str(l[0]))
    return X_all, f_all, y_all, p_all, n_all





def load_data_with_global_features(dataset_path, processed_files, gas_type):
    global_feature_path = dataset_path + f'/global_features/{gas_type}_global_features_update.csv'
    data_df = pd.read_csv(global_feature_path,header=0)
    data_x = data_df.iloc[:, 0].values
    data_f = data_df.iloc[:,1:].values.astype(np.float32)
    data_dict = {}
    for i in range(data_x.shape[0]):
        data_dict[data_x[i]] = data_f[i]
    f_all = [data_dict[_] for _ in processed_files]
    return f_all



def load_data_from_processed(dataset_path, processed_files, labels, add_dummy_node=True):
    x_all, y_all, name_all = [], [], []

    for files, label in zip(processed_files, labels):
        
        data_file = dataset_path + '/local_features/' + files + '.p'
        try:
            afm, adj, dist = pickle.load(open(data_file, "rb"))
            if add_dummy_node:
                afm, adj, dist = add_dummy_node_func(afm, adj, dist)
            x_all.append([afm, adj, dist])
            y_all.append([label])
            name_all.append(files)
        except:
            pass

    return x_all, y_all, name_all

def add_dummy_node_func(node_features, adj_matrix, dist_matrix):
    m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
    m[1:, 1:] = node_features
    m[0, 0] = 1.
    node_features = m

    m = np.ones((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
    m[1:, 1:] = adj_matrix
    adj_matrix = m

    m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
    m[1:, 1:] = dist_matrix
    dist_matrix = m

    return node_features, adj_matrix, dist_matrix


class MOF:
    def __init__(self, x, y, index, feature = None):
        self.node_features = x[0]
        self.adjacency_matrix = x[1]
        self.distance_matrix = x[2]
        self.y = y
        self.index = index
        self.global_feature = feature


class MOFDataset(Dataset):

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MOFDataset(self.data_list[key])
        return self.data_list[key]


class RealMOFDataset(Dataset):
    def __init__(self, data_list, pressure_list, ori_point):
        self.data_list = data_list
        self.pressure_list = pressure_list
        self.ori_point = np.log(np.float32(ori_point))
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self,key):
        if type(key) == slice:
            return RealMOFDataset(self.data_list[key], self.pressure_list[key], self.ori_point)
        tar_mol = copy.deepcopy(self.data_list[key])
        tar_p = np.log(self.pressure_list[key]) - self.ori_point
        tar_mol.global_feature = np.append(tar_mol.global_feature, tar_p)
        tar_mol.y = tar_mol.y
        return tar_mol

class MOFDatasetPressureVer(Dataset):

    def __init__(self, data_list, pressure_list, mask_point=None, is_train=True, tar_point=None):
        self.data_list = data_list
        self.pressure_list = pressure_list
        self.mask_point = mask_point
        self.is_train = is_train
        self.tar_point = tar_point
        if is_train:
            self.use_idx = np.where(pressure_list != mask_point)[0]
        else:
            self.use_idx = np.where(pressure_list == tar_point)[0]
        self.calcMid()

    def __len__(self):
        return len(self.data_list)

    def toStr(self):
        return {"data_list":self.data_list,"pressure_list":self.pressure_list,"mask_point":self.mask_point,"is_train":self.is_train, "tar_point":self.tar_point}
    def __getitem__(self, key):
        if type(key) == slice:
            return MOFDataset(self.data_list[key], self.pressure_list, self.mask_point, self.is_train)
        tar_mol = copy.deepcopy(self.data_list[key])
        if self.is_train:
            tar_p = self.float_pressure - self.mid
            tar_mol.global_feature = np.append(tar_mol.global_feature, tar_p)
            tar_mol.y = tar_mol.y[0]
        else:
            tar_idx = self.use_idx
            tar_p = self.float_pressure[tar_idx] - self.mid
            tar_mol.global_feature = np.append(tar_mol.global_feature, tar_p)
            tar_mol.y = [tar_mol.y[0][tar_idx]]
        return tar_mol

    def changeTarPoint(self,tar_point):
        self.tar_point = tar_point
        if not tar_point:
            self.is_train = True
        else:
            self.is_train = False
        if not self.is_train:
            self.use_idx = np.where(self.pressure_list == tar_point)[0]

    def calcMid(self):
        self.float_pressure = np.log(self.pressure_list.astype(np.float))
        self.mid = np.log(np.float(self.mask_point))


def pad_array(array, shape, dtype=np.float32):
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


def mof_collate_func_gf(batch):
    adjacency_list, distance_list, features_list, global_features_list = [], [], [], []
    labels = []

    max_size = 0
    for molecule in batch:
        if type(molecule.y[0]) == np.ndarray:
            labels.append(molecule.y[0])
        else:
            labels.append(molecule.y)
        if molecule.adjacency_matrix.shape[0] > max_size:
            max_size = molecule.adjacency_matrix.shape[0]

    for molecule in batch:
        adjacency_list.append(pad_array(molecule.adjacency_matrix, (max_size, max_size)))
        distance_list.append(pad_array(molecule.distance_matrix, (max_size, max_size)))
        features_list.append(pad_array(molecule.node_features, (max_size, molecule.node_features.shape[1])))
        global_features_list.append(molecule.global_feature)

    return [FloatTensor(features) for features in (adjacency_list, features_list, distance_list, global_features_list, labels)]  


def construct_dataset(x_all, y_all):
    output = [MOF(data[0], data[1], i)
              for i, data in enumerate(zip(x_all, y_all))]
    return MOFDataset(output)

def construct_dataset_gf(x_all, f_all, y_all):
    output = [MOF(data[0], data[2], i, data[1])
              for i, data in enumerate(zip(x_all, f_all, y_all))]
    return MOFDataset(output)

def construct_dataset_gf_pressurever(x_all, f_all, y_all, pressure_list, is_train=True, mask_point=None, tar_point=None):
    output = [MOF(data[0], data[2], i, data[1])
              for i, data in enumerate(zip(x_all, f_all, y_all))]
    return MOFDatasetPressureVer(output, pressure_list, is_train=is_train, mask_point=mask_point,tar_point=tar_point)

def construct_dataset_real(x_all, f_all, y_all, pressure_list, tar_point=None):
    output = [MOF(data[0], data[2], i, data[1])
              for i, data in enumerate(zip(x_all, f_all, y_all))]
    return RealMOFDataset(output, pressure_list, ori_point=tar_point)

def construct_loader_gf(x,f,y, batch_size, shuffle=True):
    data_set = construct_dataset_gf(x, f, y)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                         batch_size=batch_size,
					 num_workers=0,
                                         collate_fn=mof_collate_func_gf,
					 pin_memory=True,
                                         shuffle=shuffle)
    return loader

def construct_loader_gf_pressurever(data_set, batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                         batch_size=batch_size,
					 num_workers=0,
                                         collate_fn=mof_collate_func_gf,
					 pin_memory=True,
                                         shuffle=shuffle)
    return loader

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
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
