import math
import numpy as np
import bottleneck as bn
import collections


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(int(n), k)]).sum() for n in heldout_batch.sum(axis=1)])
    return DCG / IDCG

def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1) # top k
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0)
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall

def ILD(ranklist, item_cate_dict):
    K = len(ranklist)
    item_cate_arr = np.array(list(item_cate_dict.values()))
    tmp = 0
    catelist = item_cate_arr[ranklist]
    for i in range(len(catelist)):
        for j in range(i, len(catelist)):
            if catelist[i] != catelist[j]:
                tmp += 1   #两个item的category vec计算cos距离，相当于不相同category的加上1
    tmp = 2*tmp/(K*(K-1))
    return tmp

def ILD_k_batch(X_pred, item_cate_dict, k=50):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1) # top k
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    result = []
    for i in range(batch_users):
        result.append(ILD(idx[i, :k], item_cate_dict))
    res = np.array(result)
    return res

def CC(ranklist, gtItems, item_cate_dict): 
    item_cate_arr = np.array(list(item_cate_dict.values()))
    rank_catelist = set(item_cate_arr[ranklist])
    gt_catelist = set(item_cate_arr[list(gtItems)])
    result = len(rank_catelist & gt_catelist) / len(gt_catelist)
    return result

def CC_k_batch(X_pred, heldout_batch, item_cate_dict, k=50):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1) # top k
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (heldout_batch > 0)
    result = []
    for i in range(batch_users):
        result.append(CC(idx[i, :k], np.where(X_true_binary[i, :])[0], item_cate_dict))
    res = np.array(result)
    return res


def Gini_k_batch(X_pred, num_items, k=20):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1) # top k
    res = get_gini(idx[:, :k], num_items)
    return res

def get_gini(item_matrix, num_items):
    item_count = dict(collections.Counter(item_matrix.flatten()))
    sorted_count = np.array(sorted(item_count.values()))
    num_recommended_items = sorted_count.shape[0]
    total_num = item_matrix.shape[0] * item_matrix.shape[1]
    idx = np.arange(num_items - num_recommended_items + 1, num_items + 1)
    gini_index = np.sum((2 * idx - num_items - 1) * sorted_count) / total_num
    gini_index /= num_items
    return gini_index