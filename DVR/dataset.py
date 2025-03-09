import os
import pandas as pd
from tqdm import tqdm
import collections
import numpy as np
import pickle
import scipy.sparse as sp
import torch
import time
import argparse
data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Dataset(object):
    def __init__(self):
        pass
    
    def _split_A_hat(self,A, num_users, num_items):
        A_fold = []
        fold_len = (num_users + num_items) // self.opt.A_fold
        for i_fold in range(self.opt.A_fold):
            start = i_fold*fold_len
            if i_fold == self.opt.A_fold - 1:
                end = num_users + num_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce())
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def get_sparse_graph(self, train_dict, num_users, num_items, data_name):
        bipartitie_data = []
        for user, items in train_dict.items():
            for item in items:
                bipartitie_data.append([user, item])
        UI_csr = sp.csr_matrix(([1]*len(bipartitie_data), ([i[0] for i in bipartitie_data], [i[1] for i in bipartitie_data])), shape=(num_users, num_items))
        print("loading adjacency matrix")
        try:
            pre_adj_mat = sp.load_npz(os.path.join(data_dir, 'dataset', data_name,'s_pre_adj_mat.npz'))
            print("successfully loaded...")
            norm_adj = pre_adj_mat
        except :
            print("generating adjacency matrix")
            s = time.time()
            adj_mat = sp.dok_matrix((num_users + num_items, num_users + num_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = UI_csr.tolil()
            adj_mat[:num_users, num_users:] = R
            adj_mat[num_users:, :num_users] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
            
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time.time()
            print(f"costing {end-s}s, saved norm_mat...")
            sp.save_npz(os.path.join(data_dir, 'dataset', data_name,'s_pre_adj_mat.npz'), norm_adj)

        # if self.opt.A_split == True:
        #     Graph = self._split_A_hat(norm_adj, num_users, num_items)
        #     print("done split matrix")
        # else:
        Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce()
        print("don't split the matrix")
        return Graph

class Beauty(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.train_dict, self.test_dict, self.num_users, self.num_items, self.item_cate_dict, self.cate_item_dict  = self.generate_dataset()

    def generate_dataset(self):
        train_path = os.path.join(data_dir, 'dataset/beauty/beauty_data.pkl')
        dataset = pickle.load(open(train_path, 'rb'))
        train_user_list = dataset['train_user_list'][1:]
        test_user_list = dataset['test_user_list'][1:]
        item_cate_dict = dataset['item_cate_dict']

        interaction_pair = []
        for user, item_set in enumerate(train_user_list):
            for item in list(item_set):
                interaction_pair.append([user, item, item_cate_dict[item], 1])
        for user, item_set in enumerate(test_user_list):
            for item in list(item_set):
                interaction_pair.append([user, item, item_cate_dict[item], 0])
        df = pd.DataFrame(interaction_pair, columns=['user', 'item', 'cate', 'is_train'])
        df['user'] = df['user'].astype('category').cat.codes
        df['item'] = df['item'].astype('category').cat.codes
        num_users = len(df['user'].unique())
        num_items = len(df['item'].unique())

        train_df = df[df['is_train'] == 1]
        test_df = df[df['is_train'] == 0]

        train_csr = sp.csr_matrix((np.ones(train_df.shape[0]), (train_df['user'], train_df['item'])), shape=(num_users, num_items))
        test_csr = sp.csr_matrix((np.ones(test_df.shape[0]), (test_df['user'], test_df['item'])), shape=(num_users, num_items))
        
        train_dict = collections.defaultdict(list)
        test_dict = collections.defaultdict(list)
        for uid, item_list in enumerate(train_csr):
            for item in item_list.indices.tolist():
                train_dict[uid].append(item)
        
        for uid, item_list in enumerate(test_csr):
            for item in item_list.indices.tolist():
                test_dict[uid].append(item)
        item_cate_dict = df.set_index('item')['cate'].to_dict()
        cate_item_dict = df.groupby('cate')['item'].apply(list).to_dict()
        return train_dict, test_dict, num_users, num_items, item_cate_dict, cate_item_dict

class CD(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.data_path = os.path.join(data_dir, 'dataset/cd/amazon-cd-diversity.pkl')
        self.train_dict, self.test_dict, self.num_users, self.num_items, self.item_cate_dict = self.generate_dataset()

    def generate_dataset(self):
        with open(self.data_path, 'rb') as f:
            train_matrix, val_matrix, test_matrix, train_set, val_set, test_set, item_cate_dict = pickle.load(f)
        
        num_users = train_matrix.shape[0]
        num_items = train_matrix.shape[1]
        train_dict = collections.defaultdict(list)
        test_dict = collections.defaultdict(list)
        for user in range(num_users):
            train_dict[user] = list(train_set[user]) + list(val_set[user])
            test_dict[user] = list(test_set[user])
        return train_dict, test_dict, num_users, num_items, item_cate_dict


class Yelp2018(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.data_path = os.path.join(data_dir, 'dataset/yelp2018/yelp2018.pkl')
        self.train_dict, self.test_dict, self.num_users, self.num_items = self.generate_dataset()

    def generate_dataset(self):
        with open(self.data_path, 'rb') as f:
            train_matrix, val_matrix, test_matrix, train_set, val_set, test_set = pickle.load(f)
        
        num_users = train_matrix.shape[0]
        num_items = train_matrix.shape[1]
        train_dict = collections.defaultdict(list)
        test_dict = collections.defaultdict(list)
        for user in range(num_users):
            train_dict[user] = list(train_set[user]) + list(val_set[user])
            test_dict[user] = list(test_set[user])
        return train_dict, test_dict, num_users, num_items


class Gowalla(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.dir_path = os.path.join(data_dir, 'dataset/gowalla')
        self.train_file = 'train.txt'
        self.test_file = 'test.txt'
        self.train_dict, self.test_dict, self.num_users, self.num_items = self.generate_dataset()

    def generate_dataset(self):
        train_path = os.path.join(self.dir_path, self.train_file)
        test_path = os.path.join(self.dir_path, self.test_file)

        max_uid, max_iid = 0, 0
        train_dict = collections.defaultdict(list)
        with open(train_path, 'r') as f:
            train_data = f.readlines()
            for line in train_data:
                line = line.strip().split(' ')
                uid = int(line[0])
                iid_list = [int(iid) for iid in line[1:]]
                train_dict[uid] = iid_list
                max_uid = max(max_uid, uid)
                max_iid = max(max_iid, max(iid_list))
        
        test_dict = collections.defaultdict(list)
        with open(test_path, 'r') as f:
            test_data = f.readlines()
            for line in test_data:
                line = line.strip().split(' ')
                uid = int(line[0])
                iid_list = [int(iid) for iid in line[1:]]
                test_dict[uid] = iid_list
                max_uid = max(max_uid, uid)
                max_iid = max(max_iid, max(iid_list))
        
        num_users = max_uid + 1
        num_items = max_iid + 1
        return train_dict, test_dict, num_users, num_items

class LastFM(object):
    def __init__(self, opt):
        self.opt = opt
        self.data_path = os.path.join(data_dir, 'dataset/lastfm_fairness/lastfm.pkl')
        self.train_dict, self.test_dict, self.num_users, self.num_items, self.user_gender_dict, self.user_age_dict = self.generate_dataset()

    def generate_dataset(self):
        with open(self.data_path, 'rb') as f:
            train_matrix, val_matrix, test_matrix, train_set, val_set, test_set, user_gender_dict, user_age_dict = pickle.load(f)
        
        num_users = train_matrix.shape[0]
        num_items = train_matrix.shape[1]
        train_dict = collections.defaultdict(list)
        test_dict = collections.defaultdict(list)
        for user in range(num_users):
            train_dict[user] = list(train_set[user]) + list(val_set[user])
            test_dict[user] = list(test_set[user])
        return train_dict, test_dict, num_users, num_items, user_gender_dict, user_age_dict

if __name__ == '__main__':
    opt = argparse.Namespace()
    dataset1 = Beauty(opt)
    dataset2 = Yelp2018()
    dataset3 = Gowalla()
    dataset4 = LastFM()
    dataset5 = CD()
    print(dataset1.num_users)
    print(dataset1.num_items)
    print(dataset2.num_users)
    print(dataset2.num_items)
    print(dataset3.num_users)
    print(dataset3.num_items)
    print(dataset4.num_users)
    print(dataset4.num_items)
    print(dataset5.num_users)
    print(dataset5.num_items)