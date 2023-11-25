import pandas as pd
import scanpy as sc
import anndata
from pathlib import Path
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, train_test_split, KFold
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, cohen_kappa_score, average_precision_score, f1_score
from sklearn import datasets
from torchMBM.model import MBM
import time
from ikarus import utils, classifier, gene_list
import upsetplot

#save and load pickle files
def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj


def get_3selected_dataset(adatas,traindata_name,referencedata_name,testdata_name):
    """
    Obtain the gene intersection of the datasets
    
    Parameters
    ----------
    adatas
        Anndata datasets that contain training dataset, testing dataset, and reference dataset.
    traindata_name
        The name of training dataset in adatas
    referencedata_name
        The name of reference dataset in adatas
    testdata_name
        The name of test dataset in adatas
    
    Returns
    -------
    
   
    """
    
    traindata = adatas[traindata_name] 
    testdata = adatas[testdata_name] 
    referencedata = get_reference_dataset(adatas, referencedata_name) 
    adata = anndata.concat([adatas[traindata_name], adatas[testdata_name]])
    sc.pp.highly_variable_genes(adata)
    gmt = adata[:, adata.var.highly_variable].var.index.tolist()
    gene_u = np.intersect1d(gmt,adatas[referencedata_name].var_names.to_list())
    train_data =traindata[:,gene_u]
    test_data = testdata[:,gene_u]
    reference_data = referencedata[:,gene_u]
    return train_data, reference_data, test_data

def get_reference_dataset(adatas, referencedata_name):
    """
    Generate pseudo-bulk data through reference datasets
    
    Parameters
    ----------
    adatas
        Anndata datasets that contain training dataset, testing dataset, and reference dataset.
    referencedata_name
        The name of reference dataset in adatas
        
    Returns
    -------
    
   
    """
    if referencedata_name+"_ref" in adatas:
        return adatas[referencedata_name+"_ref"]
    else:
        sc.pp.neighbors(adatas[referencedata_name], n_neighbors = 10, n_pcs = 40)
        sc.tl.leiden(adatas[referencedata_name], key_added = 'leiden', random_state = 10)
        grouped_df_Tumor = adatas[referencedata_name].obs.groupby('leiden')
        grouped_data = {}

        for name, group in grouped_df_Tumor:
            sample_df = group.sample(n = min(adatas[referencedata_name].obs.leiden.value_counts()))
            result = np.sum(adatas[referencedata_name][sample_df.index,:].X.A, axis = 0)
            for i in range(99):
                sample_df = group.sample(n=min(adatas[referencedata_name].obs.leiden.value_counts()))
                result1 = np.sum(adatas[referencedata_name][sample_df.index,:].X.A, axis = 0)
                result = np.vstack((result, result1))
            result = pd.DataFrame(result, columns = adatas[referencedata_name].var_names.to_list())
            grouped_data[name] = result
        res = pd.DataFrame()
        for i in range(len(adatas[referencedata_name].obs.leiden.value_counts()) ):
            res = pd.concat([res, grouped_data[str(i)]])
            adatas[referencedata_name+"_ref"] = anndata.AnnData(res)
        return adatas[referencedata_name+"_ref"]
    
def load_data(train_data, reference_data, main_obs, batch_size, split_ratio,add_noise = True, sigma = 0.5):
    """
    (write functions here)
    
    Parameters
    ----------
    
    
    Returns
    -------
    
   
    """
    tdata_X = train_data.X.copy()
    tdata_y = train_data.obs[main_obs]
    tdata_y = [0 if i == 'Normal' else i for i in tdata_y]
    tdata_y = [1 if i == 'Tumor' else i for i in tdata_y]
    tdata_X = np.array(tdata_X.A)
    ## add some noise
    if add_noise:
        tdata_X = tdata_X + np.random.normal(0, sigma, size = tdata_X.shape)
        X_train, X_var, y_train, y_var = train_test_split(tdata_X,
                                      tdata_y,
                                          test_size = split_ratio,
                                          random_state = 42)

    X_train, y_train = torch.as_tensor(X_train).float(), torch.as_tensor(y_train).float()
    X_var, y_var = torch.as_tensor(X_var).float(), torch.as_tensor(y_var).float()

    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(ds_train, batch_size = batch_size, shuffle = True)

    ds_var = torch.utils.data.TensorDataset(X_var, y_var)
    validloader = torch.utils.data.DataLoader(ds_var, batch_size = batch_size, shuffle = True)

    return trainloader, validloader

def intersection_fun(x):
    return pd.concat(x, axis=1, join="inner")

def union_fun(x):
    return pd.concat(x,axis=1, join="outer")

def get_genelist(adatas, traindata_name, save_path, obs_name='sub_ct'):
    """
    select a signature list using ikarus's method.
    
    Parameters
    ----------
    adatas
        Anndata datasets that contain training dataset, testing dataset, and reference dataset.
    traindata_name
        The name of training dataset in adatas
    save_path
        The path where the gene list is saved
    obs_name
        The label of cells used to select the gene list.

    """
    train_data = adatas[traindata_name]
    cluster_list = list(set(train_data.obs[obs_name]))
    if 'None' in cluster_list:
        cluster_list.remove('None')
    if 'nan' in cluster_list:
        cluster_list.remove('nan')
    cluster_list.remove('Tumor')
    
    label_upregs = ["Tumor"]*len(cluster_list)
    label_downregs = cluster_list
    tumor_signatures = gene_list.create_all(label_upregs, label_downregs, adatas, 
                                        [traindata_name], [obs_name], intersection_fun)
    contents = upsetplot.from_contents(tumor_signatures)
    x = tuple([True]*(len(cluster_list)))
    tumor_genes = contents.loc[x].values.ravel().tolist()
    normal_signatures = gene_list.create_all(
        label_upregs_list=label_downregs,
        label_downregs_list=label_upregs,
        adatas_dict=adatas,
        names_list=[traindata_name],
        obs_names_list=[obs_name],
        integration_fun=intersection_fun,
        top_x=300
    )
    tumor_genes_union = []
    for i in tumor_signatures.values():
        tumor_genes_union += i
    tumor_genes_union = list(set(tumor_genes_union)) # unique genes
    normal_genes_union = []
    for i in normal_signatures.values():
        normal_genes_union += i
    normal_genes_union = list(set(normal_genes_union))
    overlap = list(set(tumor_genes_union) & set(normal_genes_union))
    normal_genes = list(set(normal_genes_union) - set(overlap))
    
    gene_lists = [normal_genes, tumor_genes]
    gene_list_names = ["Normal", "Tumor"]
    gmt = pd.DataFrame(gene_lists, index=gene_list_names)
    gmt.insert(0, "00", "ikarus")
    gmt.to_csv(save_path + traindata_name + "/signatures.gmt", header=None, sep="\t")

