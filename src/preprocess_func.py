import os
import numpy as np
import matplotlib.pyplot as plt
import copy

from tqdm import tqdm_notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import seaborn

import pymatgen
from pymatgen import core
import pandas as pd

from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
import math

import json
from collections import defaultdict


class data_generator(object):
  """ Creates a generator object from the compositions provided"""
  def __init__(self, comps):
        all_eles = []
        for c in comps:
            all_eles += list(c.get_el_amt_dict().keys())
        all_eles += ['O']
        self.eles = np.array(sorted(list(set(all_eles))))

        self.elements = self.eles
        self.size = len(self.eles)
        self.length = len(comps)

        all_vecs = np.zeros([len(comps), len(self.elements)])
        for i, c in enumerate(comps):
            for k, v in c.get_el_amt_dict().items():
                j = np.argwhere(self.eles == k)
                all_vecs[i, j] = v
        all_vecs = all_vecs / np.sum(all_vecs, axis=1).reshape(-1, 1)
        self.real_data = np.array(all_vecs, dtype=np.float32)

  def sample(self, N):
      """ Randomly sample compositions"""
      idx = np.random.choice(np.arange(self.length), N, replace=False)
      data = self.real_data[idx]

      return np.array(data, dtype=np.float32),idx
    
  def elements(self):
      """ Return all the elements present in the dataset"""
      return self.eles

def structure_choose(metaIndex_dict,meta_dict,material,n_index):
    metaIndex_update1={}
    for j,k in metaIndex_dict.items():
        comb_final=0
        data={}
        comb=0
        sum_comb=0
        if k>2:
            for o in material['compositionDictionary'].keys():
                if o not in meta_dict:
                    data[j] = None
                    break

                structure = ''
                if n_index != None:
                    s = n_index
                    structure = material['structure'][s]
                else:
                    try:
                        for a in meta_dict[o].keys():
                            int(a)
                            structure = meta_dict[o][a][1]
                    except:
                        structure='BCC'
                    #print('strucutre',material['compositionDictionary'],o,structure)

                try:
                    data[j]=meta_dict[o][structure][k]
                    float(data[j])
                    ##print('BCC')
                except:
                    ##print('No vaule for '+j+' of '+o+' for the phase in records, try other structures')
                    if structure=='BCC':
                        try:
                            data[j]=meta_dict[o]['FCC'][k]
                            float(data[j])
                            ##print('FCC')
                        except:
                            try:
                                data[j]=meta_dict[o]['HCP'][k]
                                float(data[j])
                                ##print('HCP')
                            except:
                                ##print('BREAk')
                                data[j] = None
                                break
    
                    elif structure=='FCC':
                        try:
                            data[j]=meta_dict[o]['HCP'][k]
                            float(data[j])
                        except:
                            try:
                                data[j]=meta_dict[o]['BCC'][k]
                                float(data[j])
                            except:
                                data[j] = None
                                break
                    elif structure=='HCP':
                        try:
                            data[j]=meta_dict[o]['FCC'][k]
                            float(data[j])
                        except:
                            try:
                                data[j]=meta_dict[o]['BCC'][k]
                                float(data[j])
                            except:
                                data[j] = None
                                break
                    elif structure=='Others':
                        try:
                            data[j]=meta_dict[o]['BCC'][k]
                            float(data[j])
                            #print('others','BCC')
                        except:
                            try:
                                data[j]=meta_dict[o]['FCC'][k]
                                float(data[j])
                                #print('others','FCC')
                            except:
                                try:
                                    data[j]=meta_dict[o]['HCP'][k]
                                    float(data[j])
                                    #print('others','HCP')
                                except:
                                    data[j] = None
                                    break
                

                ##print('data',comb,i['material']['compositionDictionary'][o])
                comb=comb+material['compositionDictionary'][o]*data[j]
                
                sum_comb=sum_comb+material['compositionDictionary'][o]
                        
            if data[j] != None:
                metaIndex_update1[j] = round(float(comb/sum_comb),6)
            else:
                metaIndex_update1[j] = None

    ##print(metaIndex_update1)
    return metaIndex_update1

def structure_calculate(metaIndex_dict,meta_dict,material):
    all_structure=['BCC','FCC','HCP']
    result=[]
    n = 0
    n_index=[]
    try:
        for i in range(len(material['structure'])):
            if material['structure'][i] in all_structure:
                n=n+1
                n_index.append(i) 
        #print('n and st',n)
    except:
        pass
    if n>1:
        #print('n_value',material['formula'],material['structure'][n_index[0]])
        for s in n_index:
            singleResult = structure_choose(metaIndex_dict,meta_dict,material,s)
            singleResult['structure'] = material['structure'][s]
            result.append(singleResult)
    elif n==1:
        #print('n_value1',material['formula'],material['structure'][n_index[0]])
        singleResult = structure_choose(metaIndex_dict,meta_dict,material,n_index[0])
        singleResult['structure'] = material['structure'][n_index[0]]
        result.append(singleResult)
    elif n==0:
        #print('n_value0',material['formula'])
        singleResult = structure_choose(metaIndex_dict, meta_dict, material, None)
        singleResult['structure'] = '?'
        result.append(singleResult)
    ##print(metaIndex_update)
    return result

def linear_combination_run(data):
    excelFile = "..\srf\FundemantalDescriptors_PureElements.xlsx"
    metaDF = pd.read_excel(excelFile)
    meta = metaDF.to_json(orient="split")
    metaIndex = json.loads(meta)['columns']
    metaParsed = json.loads(meta)['data']
    meta_dict=defaultdict(dict)
    metaIndex_dict={}
    for k in metaParsed:
        meta_dict[k[2]][k[0]]=k
    ##print(metaIndex,meta_dict)
    for j in range(len(metaIndex)):
        metaIndex_dict[metaIndex[j]] = j
    result = structure_calculate(metaIndex_dict,meta_dict,data['material'])
    return result

def FT_Rice_92(Shear_Modulus,Unstable_Stacking_Fault_Energy,Poisson_Ratio ):
    if None in [Unstable_Stacking_Fault_Energy, Shear_Modulus, Poisson_Ratio]:
        return None
    else:
        return math.sqrt(2*Shear_Modulus*Unstable_Stacking_Fault_Energy/(1-Poisson_Ratio))

def get_sm_ft_vals(compositions):
  """ get the shear modulus and fracture toughness values from compositions in Pymatgen format"""
  ft_val_list = []
  sm_val_list = []
  ks = {}
  for i in range(len(compositions)):
    temp = {}
    temp['formula'] = compositions[i].formula
    temp['compositionDictionary']=compositions[i].as_dict()
    temp['reducedFormula']= compositions[i].reduced_formula
    temp['structure']=['?']
    ks['material']=temp
    LCR = linear_combination_run(ks)
    ft_val = FT_Rice_92(LCR[0]['DFTGh'],LCR[0]['USFE'],LCR[0]['DFTpoisson'])
    ft_val_list.append(ft_val)
    sm_val_list.append(LCR[0]['DFTGh'])
  return sm_val_list,ft_val_list

def fig_plot(points, dset, thresh=0.03):

    fig, axes = plt.subplots(1, 4, figsize=(25, 5))
    ax=axes[0]
    ax.violinplot(points, np.arange(points.shape[1]))
    ax.set_xticks(np.arange(dset.size))
    ax.set_xticklabels(dset.elements,rotation=45)
    ax.set_xlabel('Elements')
    ax.set_ylabel('Frequency')
    s = points.sum(axis=1)
    ax=axes[1]
    if np.max(s) - np.min(s) < 0.1:
        ax.hist(s, np.linspace(0.9, 1.1, 6))
    else:
        ax.hist(s)
    ax.set_xlabel('Sum of fractions')
    ax.set_ylabel('Frequency')
    ax = axes[2]
    ax.hist(points.flatten())
    ax.set_xlabel('Elemental fraction')
    ax.set_ylabel('Frequency')
    ne = (points > thresh).sum(axis=1)
    ax = axes[3]
    ax.hist(ne, np.arange(0, 17), width=0.80)
    ax.set_xlabel('Number of elements')
    ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    plt.close()
