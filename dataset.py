import torch 
import torch.nn as nn
import copy
from random import sample
import scipy.sparse as sparse
from tqdm import tqdm
from kg_completion import parse_rdf
"""
Dataset class
"""
class Dataset(object):
    def __init__(self, data_root, sparsity=1, inv=True):
        # Construct entity_list
        entity_path = data_root + 'entities.txt'
        self.idx2ent_, self.ent2idx_ = load_entities(entity_path)
        # 获取实体数量
        self.entity_num = len(self.idx2ent_)
        # Construct rdict which contains relation2idx & idx2relation2 
        relation_path = data_root + 'relations.txt'
        self.rdict = Dictionary()
        self.load_relation_dict(relation_path)
        # head relation
        self.head_rdict = Dictionary()
        self.head_rdict = copy.deepcopy(self.rdict)
        # load (h, r, t) tuples
        fact_path     = data_root + 'facts.txt'
        train_path    = data_root + 'train.txt'
        valid_path    = data_root + 'valid.txt'
        test_path     = data_root + 'test.txt'
        if inv :
            fact_path += '.inv'
        self.rdf_data_ = self.load_data_(fact_path, train_path, valid_path, test_path, sparsity)
        self.fact_rdf_, self.train_rdf_, self.valid_rdf_, self.test_rdf_ = self.rdf_data_
        # inverse
        if inv :
            # add inverse relation to rdict
            rel_list = list(self.rdict.rel2idx_.keys())
            for rel in rel_list:
                inv_rel = "inv_" + rel
                self.rdict.add_relation(inv_rel)                
                self.head_rdict.add_relation(inv_rel)                
        # add None 
        self.head_rdict.add_relation("None")
        self.r2mat = self.construct_rmat(self.rdict.idx2rel,self.idx2ent_, self.ent2idx_, self.fact_rdf_+ self.train_rdf_+self.valid_rdf_)


    def load_rdfs(self, path):
        rdf_list = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                tuples = line.strip().split('\t')
                rdf_list.append(tuples)
        return rdf_list
    def construct_rmat(self,idx2rel, idx2ent, ent2idx, fact_rdf):
        e_num = len(idx2ent)
        r2mat = {}
        for idx, rel in idx2rel.items():
            mat = sparse.dok_matrix((e_num, e_num))
            r2mat[rel] = mat
        # fill rmat
        for rdf in tqdm(fact_rdf, desc="constructing rmat"):
            fact = parse_rdf(rdf)
            h, r, t = fact
            h_idx, t_idx = ent2idx[h], ent2idx[t]
            r2mat[r][h_idx, t_idx] = 1
        for rel, mat in r2mat.items():
            r2mat[rel] = mat.tocsr()
        return r2mat   
    def load_data_(self, fact_path, train_path, valid_path, test_path, sparsity):
        fact  = self.load_rdfs(fact_path)
        fact = sample(fact ,int(len(fact)*sparsity))
        train = self.load_rdfs(train_path)
        valid = self.load_rdfs(valid_path)
        test  = self.load_rdfs(test_path)
        return fact, train, valid, test
    
    def load_relation_dict(self, relation_path):
        """
        Read relation.txt to relation dictionary
        """
        with open(relation_path, encoding='utf-8') as f:
            rel_list = f.readlines()
            for r in rel_list:
                relation = r.strip()
                self.rdict.add_relation(relation)
                #self.head_dict.add_relation(relation)
    
    def get_relation_dict(self):
        return self.rdict
    
    def get_head_relation_dict(self):
        return self.head_rdict

    @property
    def idx2ent(self):
        return self.idx2ent_

    @property
    def ent2idx(self):
        return self.ent2idx_

    @property
    def fact_rdf(self):
        return self.fact_rdf_
    
    @property
    def train_rdf(self):
        return self.train_rdf_
    
    @property
    def valid_rdf(self):
        return self.valid_rdf_
    
    @property
    def test_rdf(self):
        return self.test_rdf_
    
"""
Dictionary class
"""

class Dictionary(object):
    def __init__(self):
        self.rel2idx_ = {}
        self.idx2rel_ = {}
        self.idx = 0
        
    def add_relation(self, rel):
        if rel not in self.rel2idx_.keys():
            self.rel2idx_[rel] = self.idx
            self.idx2rel_[self.idx] = rel
            self.idx += 1
        
    @property
    def rel2idx(self):
        return self.rel2idx_
    
    @property
    def idx2rel(self):
        return self.idx2rel_
    
    def __len__(self):
        return len(self.idx2rel_)

def load_entities(path):
    idx2ent, ent2idx = {}, {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            e = line.strip()
            ent2idx[e] = idx
            idx2ent[idx] = e
    return idx2ent, ent2idx  