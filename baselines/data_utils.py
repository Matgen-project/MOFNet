"""

"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
#torch.multiprocessing.set_start_method('spawn')
#from rdkit import Chem
#from rdkit.Chem import AllChem
#from rdkit.Chem import MolFromSmiles
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset, dataset
from scipy.sparse import coo_matrix
import json
import copy


#use_cuda = torch.cuda.is_available()
use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

def load_data_from_df(dataset_path, gas_type, pressure, use_global_features=False):
    """Load and featurize data stored in a CSV file.

    Args:
        dataset_path (str): A path to the CSV file containing the data. It should have two columns:
                            the first one contains SMILES strings of the compounds,
                            the second one contains labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    print(dataset_path + f'/label/{gas_type}/{gas_type}_ads_all.csv')
    data_df = pd.read_csv(dataset_path + f'/label/{gas_type}/{gas_type}_ads_all.csv',header=0)
    
    data_x = data_df['name'].values
    #print(pressure)
    #print(data_df[pressure])
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
    #print(global_feature_path)
    data_df = pd.read_csv(global_feature_path,header=0)
    data_x = data_df.iloc[:, 0].values
    data_f = data_df.iloc[:,1:].values.astype(np.float32)
    data_dict = {}
    for i in range(data_x.shape[0]):
        data_dict[data_x[i]] = data_f[i]
    f_all = [data_dict[_] for _ in processed_files]
    return f_all



def load_data_from_processed(dataset_path, processed_files, labels):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        data_set_path (list(str)): data path for processed files.
        processed_files (list(str)): processed files name.
        labels (list[float]): A list of the corresponding labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    x_all, y_all, name_all = [], [], []

    for files, label in zip(processed_files, labels):
        
        data_file = dataset_path + '/processed_en/' + files + '.p'
        #print(data_file)
        try:
            afm, row, col, pos = pickle.load(open(data_file, "rb"))
            #print(afm,adj,dist)
            x_all.append([afm, row, col, pos])
            y_all.append([label])
            name_all.append(files)
        except:
            pass

    return x_all, y_all, name_all

class Molecule:
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """

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


class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of Molecule objects
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataset(self.data_list[key])
        return self.data_list[key]

def construct_dataset_gf(x_all, f_all, y_all):
    """Construct a MolDataset object from the provided data.

    Args:
        x_all (list): A list of atom features of molecules.
        f_all (list): A list of molecule global feature.
        y_all (list): A list of the corresponding labels.

    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Molecule(data[0], data[2], i, data[1])
              for i, data in enumerate(zip(x_all, f_all, y_all))]
    return MolDataset(output)

def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array

def mol_collate_func_adj(batch):
    """Create a padded batch of molecule features with additional global features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, global features and labels.
    """
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

def mol_collate_func_nbh(batch):
    """Create a padded batch of molecule features with additional global features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, global features and labels.
    """
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
    """Construct a data loader for the provided data.

    Args:
        x (list): A list of atom features of molecules.
        f (list): A list of molecule global features.
        y (list): A list of the corresponding labels.
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    data_set = construct_dataset_gf(x, f, y)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                         batch_size=batch_size,
					                     num_workers=8,
                                         collate_fn=mol_collate_func_adj if use_adj else mol_collate_func_nbh,
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