import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
last_dir = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(last_dir)
import torch
import argparse
import logging
import numpy as np
import time
from torch.distributions import binomial
import random
import json


from dataloader import DataLoader
from backbone.bpr import BPR
from value import Value
from eval_metrics import NDCG_binary_at_k_batch, Recall_at_k_batch
from eval_metrics import CC_k_batch, ILD_k_batch, Gini_k_batch
from dataset import Beauty, CD, Yelp2018, Gowalla, LastFM
from harsanyi import HarsanyiNet



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, required=True)
parser.add_argument('-d', '--dataset', type=str, choices=['beauty', 'cd', 'yelp2018', 'lastfm', 'gowalla'], required=True)
parser.add_argument('--reward', type=str, choices=['loss', 'recall', 'ndcg', 'cc', 'ild', 'gini', 'age_eo', 'age_sp', 'gender_eo', 'gender_sp'], default='loss')
parser.add_argument('--backbone', type=str, choices=['bpr'], default='bpr')
parser.add_argument('--num_users', type=int, default=0)
parser.add_argument('--num_items', type=int, default=0)
parser.add_argument('--num_interactions', type=int, default=0)
parser.add_argument('--user_dim', type=int, default=64)
parser.add_argument('--item_dim', type=int, default=64)

known_args, _ = parser.parse_known_args()

hyper_params = json.load(open(os.path.join(cur_dir, 'params', f'{known_args.backbone}_{known_args.dataset}_{known_args.reward}.json')))
parser.add_argument('--valid_interval', type=int, default=10)
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--bpr_lr', type=float, default=0.001)
parser.add_argument('--dve_lr', type=float, default=0.001)
parser.add_argument('--bpr_weight_decay', type=float, default=0.001)
parser.add_argument('--dve_weight_decay', type=float, default=0.001)
parser.add_argument('--dve_update_interval', type=int, default=1)
parser.add_argument('--dve_print_interval', type=int, default=100)
parser.add_argument('--weight_mode', type=str, choices=['sample', 'dot'], default='sample')
parser.add_argument('--h_dim', type=int, default=10)
parser.add_argument('--gamma', type=int, default=100)
parser.add_argument('--mse_alpha', type=float, default=1.0)
parser.add_argument('--pg_alpha', type=float, default=1.0)
parser.add_argument('--layers', type=list, default=[128, 64, 32, 16, 8])
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--A_fold', type=int, default=100)
parser.add_argument('--l2', type=float, default=1e-4)
parser.add_argument('--layer_size', nargs='?', default='[64]', help='Output sizes of every layer')
parser.set_defaults(**hyper_params)
opt = parser.parse_args()
opt.load_path = os.path.join(last_dir, 'dataset', 'pretrained_model', f'{opt.backbone}_{opt.dataset}.pt')

seed = 3333
if opt.cuda:
    torch.cuda.manual_seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(f'cuda:{opt.cuda}')
opt.device = device

pid = os.getpid()
logger.info('pid: {}'.format(pid))

t0 = time.time()
logger.info(time.strftime("%Y-%m-%d", time.localtime(t0)))

logger.info(opt)

if opt.dataset == 'beauty':
    dataset = Beauty(opt)
elif opt.dataset == 'cd':
    dataset = CD(opt)
elif opt.dataset == 'yelp2018':
    dataset = Yelp2018(opt)
elif opt.dataset == 'lastfm':
    dataset = LastFM(opt)
elif opt.dataset == 'gowalla':
    dataset = Gowalla(opt)


if opt.reward == 'cc' or opt.reward == 'ild':
    item_cate_dict = dataset.item_cate_dict
if opt.reward in ['age_eo', 'age_sp']:
    user_age_dict = dataset.user_age_dict
if opt.reward in ['gender_eo', 'gender_sp']:
    user_gender_dict = dataset.user_gender_dict

dataloader = DataLoader(opt, dataset)
train_dataloader = dataloader.train_dataloader
test_dataloader = dataloader.test_dataloader
valid_dataloader = dataloader.valid_dataloader

opt.num_users = dataset.num_users
opt.num_items = dataset.num_items


model = BPR(opt).to(device)

if opt.load_model:
    logger.info('load model from {}'.format(opt.load_path))
    load_dict = torch.load(opt.load_path, map_location=device)
    model.load_state_dict(load_dict)

start_epoch = 0

optimizer = torch.optim.Adam(model.parameters(), lr=opt.bpr_lr, weight_decay=opt.bpr_weight_decay)

harsanyi_net = HarsanyiNet(input_dim=opt.batch_size * opt.h_dim, num_classes=1, num_layers=2, act_ratio=0.01, device=device, gamma=1000)
data_estimator = Value(opt).to(device)

dve_optimizer = torch.optim.Adam([{'params': data_estimator.parameters(), 'params': harsanyi_net.parameters()}], lr=opt.dve_lr, weight_decay=opt.dve_weight_decay)

logger.info('start training')
best_valid_eval = [0.0, 0.0, 0.0, 0.0]
# 
loss_reward = []
dve_sel_prob = []
dve_data_value = []
reward_preds = []
for epoch_num in range(start_epoch, opt.epoch):
    t1 = time.time()
    loss_log = {'bpr_loss_weighted': 0.0, 'bpr_loss_ori': 0.0, 'reward':0.0}
    recall_10_list = [0.0]
    train_dataloader.dataset.neg_sampling()
    for i, batch in enumerate(train_dataloader):
        model.train()
        batch_user, batch_item, batch_neg_item = batch
        batch_user, batch_item, batch_neg_item = batch_user.to(device), batch_item.to(device), batch_neg_item.to(device)
        targets_prediction = model(batch_user, batch_item)
        negatives_prediction = model(batch_user, batch_neg_item)
        bpr_loss = -torch.log(torch.sigmoid(targets_prediction - negatives_prediction) + 1e-8)
        

        batch_user_embedding = model.user_embedding(batch_user).detach().clone()
        batch_item_embedding = model.item_embedding(batch_item).detach().clone()
        batch_neg_item_embedding = model.item_embedding(batch_neg_item).detach().clone()

        loss_feature = bpr_loss.detach().clone().unsqueeze(1)

        batch_embedding = data_estimator(batch_user_embedding, batch_item_embedding, batch_neg_item_embedding, loss_feature)

        harsanyi_input = batch_embedding.reshape(1, -1)
        harsanyi_shapley, reward_pred = harsanyi_net.compute_shapley_tensor(harsanyi_input, device)
        harsanyi_shapley = harsanyi_shapley.reshape(opt.h_dim, opt.batch_size).sum(dim=0)    

        data_value = (harsanyi_shapley - torch.min(harsanyi_shapley))

        data_value = data_value * opt.gamma

        dve_data_value.append(data_value) 
        reward_preds.append(reward_pred)

        if opt.weight_mode == 'sample':
            distribution = binomial.Binomial(total_count=1, probs=data_value)
            sel_prob_curr = distribution.sample()
            dve_sel_prob.append(sel_prob_curr)
            bpr_loss_weighted = torch.sum(bpr_loss * sel_prob_curr)
        elif opt.weight_mode == 'dot':
            weight = data_value.detach().clone()
            bpr_loss_weighted = torch.sum(bpr_loss * weight)

        optimizer.zero_grad()
        bpr_loss_weighted.backward()
        optimizer.step()

        bpr_loss_ori = torch.sum(bpr_loss)
        bpr_loss_avg = torch.mean(bpr_loss)        
        loss_reward.append(- bpr_loss_ori.item())

        if (epoch_num * len(train_dataloader)  + i) % opt.dve_update_interval == 0:
            if opt.reward in ['recall', 'ndcg', 'cc', 'ild', 'gini', 'age_eo', 'age_sp', 'gender_eo', 'gender_sp']:
                model.eval()
                for idx, batch_test_data in enumerate(valid_dataloader):
                    batch_user_ids, batch_test_items, batch_user_items = batch_test_data
                    batch_user_ids = batch_user_ids.to(device)
                    rating_pred = model.predict(batch_user_ids)
                    rating_pred = rating_pred.reshape(-1, rating_pred.shape[-1])
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    rating_pred[batch_user_items > 0] = -np.inf
                    if idx == 0:
                        pred_list = rating_pred
                        test_matrix = np.array(batch_test_items)
                    else:
                        pred_list = np.append(pred_list, rating_pred, axis=0)
                        test_matrix = np.append(test_matrix, batch_test_items, axis=0)
                r_10 = Recall_at_k_batch(pred_list, test_matrix, 10).mean()
                n_10 = NDCG_binary_at_k_batch(pred_list, test_matrix, 10).mean()
            if opt.reward == 'recall':
                reward = Recall_at_k_batch(pred_list, test_matrix, 10).mean()
                mse_reward = torch.tensor([reward * 100], dtype=torch.float32).to(device)
                pg_reward = torch.tensor([reward], dtype=torch.float32).to(device)
            elif opt.reward == 'ndcg':
                reward = NDCG_binary_at_k_batch(pred_list, test_matrix, 10).mean()
                mse_reward = torch.tensor([reward * 100], dtype=torch.float32).to(device)
                pg_reward = torch.tensor([reward], dtype=torch.float32).to(device)
            elif opt.reward == 'loss':
                reward = np.mean(loss_reward)
                mse_reward = torch.tensor(loss_reward, dtype=torch.float32).to(device)
                pg_reward = torch.tensor(loss_reward).mean()
            elif opt.reward == 'cc':
                reward = CC_k_batch(pred_list, test_matrix, item_cate_dict, 20).mean()
                mse_reward = torch.tensor([reward], dtype=torch.float32).to(device)
                pg_reward = torch.tensor([reward], dtype=torch.float32).to(device)
            elif opt.reward == 'ild':
                reward = ILD_k_batch(pred_list, item_cate_dict, 20).mean()
                mse_reward = torch.tensor([reward], dtype=torch.float32).to(device)
                pg_reward = torch.tensor([reward], dtype=torch.float32).to(device)
            elif opt.reward == 'gini':
                reward = Gini_k_batch(pred_list, 20)
                mse_reward = torch.tensor([reward], dtype=torch.float32).to(device)
                pg_reward = torch.tensor([reward], dtype=torch.float32).to(device)
            data_estimator.train()
            harsanyi_net.train()
            
            mse_loss_fn = torch.nn.MSELoss()
            dve_mse_loss = mse_loss_fn(mse_reward, torch.concat(reward_preds))

            prob = torch.mean(torch.log(torch.concat(dve_data_value, dim=0) + 1e-8))
            dve_pg_loss = pg_reward * prob

            dve_loss = opt.mse_alpha * dve_mse_loss + opt.pg_alpha * dve_pg_loss
            dve_optimizer.zero_grad()
            dve_loss.backward()
            dve_optimizer.step()
            loss_reward = []
            dve_data_value = []
            dve_sel_prob = []
            harsanyi_outputs = []
            reward_preds = []


        loss_log['bpr_loss_weighted'] += bpr_loss_weighted.item()
        loss_log['bpr_loss_ori'] += bpr_loss_ori.item()
        loss_log['reward'] += reward

        if (epoch_num * len(train_dataloader)  + i) % opt.dve_print_interval == 0:
            logger.info(f'harsanyi_shapley max: {torch.max(harsanyi_shapley).item():3f}, mean: {torch.mean(harsanyi_shapley).item():3f}, min: {torch.min(harsanyi_shapley).item():3f}')
            format_str = 'Epoch {:3d} iter {:3d} max_weight {:.4f} min_weight {:.4f} mean_weight {:.4f}'
            logger.info(format_str.format(epoch_num, i, torch.max(data_value).item(), torch.min(data_value).item(), torch.mean(data_value).item()))
            logger.info(f'bpr_loss: {bpr_loss_ori.item():3f}, dve_loss: {dve_loss.item():3f}, mse_loss: {dve_mse_loss.item():3f}, pg_loss: {dve_pg_loss.item():3f}')

    for key in loss_log.keys():
        loss_log[key] /= len(train_dataloader)

    t2 = time.time()
    format_str = 'Epoch {:3d} [{:.1f} s]: bpr_loss_weighted={:.4f}, bpr_loss_ori={:.4f}, reward={:.4f}'
    logger.info(format_str.format(epoch_num, t2 - t1, loss_log['bpr_loss_weighted'], loss_log['bpr_loss_ori'], loss_log['reward']))

    if epoch_num > 0 and epoch_num % opt.valid_interval == 0:
        model.eval()
        for idx, batch_test_data in enumerate(test_dataloader):
            batch_user_ids, batch_test_items, batch_user_items = batch_test_data
            batch_user_ids = batch_user_ids.to(device)
            rating_pred = model.predict(batch_user_ids)
            rating_pred = rating_pred.reshape(-1, rating_pred.shape[-1])
            rating_pred = rating_pred.cpu().data.numpy().copy()
            rating_pred[batch_user_items > 0] = -np.inf
            if idx == 0:
                pred_list = rating_pred
                test_matrix = batch_test_items
            else:
                pred_list = np.append(pred_list, rating_pred, axis=0)
                test_matrix = np.append(test_matrix, batch_test_items, axis=0)
        r_10 = Recall_at_k_batch(pred_list, test_matrix, 10).mean()
        n_10 = NDCG_binary_at_k_batch(pred_list, test_matrix, 10).mean()
        r_20 = Recall_at_k_batch(pred_list, test_matrix, 20).mean()
        n_20 = NDCG_binary_at_k_batch(pred_list, test_matrix, 20).mean()

        metrics = []
        metrics.append("Recall@10 {:.5f}".format(r_10))
        metrics.append("NDCG@10 {:.5f}".format(n_10))
        metrics.append("Recall@20 {:.5f}".format(r_20))
        metrics.append("NDCG@20 {:.5f}".format(n_20))
        if opt.reward in ['cc', 'ild']:
            cc_20 = CC_k_batch(pred_list, test_matrix, item_cate_dict, 20).mean()
            ild_20 = ILD_k_batch(pred_list, item_cate_dict, 20).mean()
            metrics.append("CC@20 {:.5f}".format(cc_20))
            metrics.append("ILD@20 {:.5f}".format(ild_20))
        elif opt.reward == 'gini':
            gini = Gini_k_batch(pred_list, 20)
            metrics.append("Gini {:.5f}".format(gini))
        logger.info('\nvalid epoch {} '.format(epoch_num) + " ".join(metrics))
        logger.info("Evaluation time:{}".format(time.time() - t2))
        

        if r_20 > best_valid_eval[2]:
            best_valid_eval = [r_10, n_10, r_20, n_20]

metrics = []
metrics.append("Recall@10 {:.5f}".format(best_valid_eval[0]))
metrics.append("NDCG@10 {:.5f}".format(best_valid_eval[1]))
metrics.append("Recall@20 {:.5f}".format(best_valid_eval[2]))
metrics.append("NDCG@20 {:.5f}".format(best_valid_eval[3]))
logger.info('\nBest valid' + " ".join(metrics))

logger.info("Time cost: {:.2f}".format((time.time() - t0) / 3600))