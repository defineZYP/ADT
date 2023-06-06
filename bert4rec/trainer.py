import os
import time
import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader

from model.bert import BertModel
from datasets.dataset import data_partition, BertTrainDataset, BertEvalDataset
from datasets.negative_sampler import PopularSampler, RandomSampler

class BertTrainer:
    def __init__(self, args):
        # get dataset
        self.usernum, self.itemnum, self.train_loader, self.val_loader, self.test_loader = \
            self._get_datainfo_loader(args)
        # get model
        self.model = BertModel(
            usernum=self.usernum,
            itemnum=self.itemnum,
            args=args
        )
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                module.weight.data.normal_(mean=0.01, std=args.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.model.to(args.device)
        
        # get optimizer
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        self.num_epochs = args.num_epochs
        self.eval_interval = args.eval_interval
        self.metric_ks = [5, 10]
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.clip = args.clip
        self.args = args

    def evaluate(self, mode='val'):
        '''
        evaluate model on validation set or test set
        '''
        NDCG = {k: 0.0 for k in self.metric_ks}
        HT = {k: 0.0 for k in self.metric_ks}
        valid_user = 0.0
        loader = self.val_loader if mode == 'val' else self.test_loader
        ranks_all = []
        with tqdm.tqdm(loader) as t:
            t.set_description(f"Evaluate {mode}: ")
            for batch, _ in t:
                u, seq, item_idx = batch
                # u, seq = np.array(u), np.array(seq)
                # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
                seq_pos_ids = torch.LongTensor(np.tile(np.array(range(seq.shape[1])), [seq.shape[0], 1])).to(self.args.device)
                seq_sent_ids = torch.zeros(seq.shape, dtype=int).to(self.args.device)
                predictions = -self.model.predict(u, seq, seq_pos_ids, seq_sent_ids, item_idx)
                # for given metric, calculate it 
                # batch_size * 1
                rank = predictions.argsort(dim=1).argsort(dim=1)[:, 0]
                ranks_all.extend(rank.cpu().detach().numpy())
                valid_user += u.shape[0]
                for k in self.metric_ks:
                    rank_matrix = torch.where(rank < k)
                    hit_matrix = rank[rank_matrix]
                    ndcg_matrix = 1 / torch.log2(hit_matrix + 2)
                    HT[k] += hit_matrix.shape[0]
                    NDCG[k] += ndcg_matrix.sum()
        for k in self.metric_ks:
            HT[k] = HT[k] / valid_user
            NDCG[k] = NDCG[k].item() / valid_user
        # AUC
        ranks_all = np.array(ranks_all) + 1
        candidates_size = 1 + item_idx.shape[1]
        AUC = np.mean((candidates_size - ranks_all) / (candidates_size - 1))
        return (NDCG, HT), AUC

    def train(self, lambda1, lambda2):
        '''
        train model with lambdas
        '''
        T = 0.0
        t0 = time.time()

        best_valid_score = 0.0
        best_epoch = 0
        rec_valid = []
        rec_test = []
        rec_valid_AUC = 0
        rec_test_AUC = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            with tqdm.tqdm(self.train_loader) as t:
                for batch, labels in t:
                    t.set_description(f"Epoch: {epoch}/{self.num_epochs}")
                    seq, deq = batch
                    seq_pos_ids = torch.LongTensor(np.tile(np.array(range(seq.shape[1])), [seq.shape[0], 1])).to(self.args.device)
                    seq_sent_ids = torch.zeros(seq.shape, dtype=int).to(self.args.device)
                    deq_pos_ids = torch.LongTensor(np.tile(np.array(range(deq.shape[1])), [deq.shape[0], 1])).to(self.args.device)
                    deq_sent_ids = torch.zeros(seq.shape, dtype=int).to(self.args.device)
                    # seq, deq = np.array(seq), np.array(deq)
                    self.optimizer.zero_grad()
                    logits, encoder_layer_input, decoder_layer_output, rec_layer_hsic = self.model(seq, deq, seq_pos_ids, seq_sent_ids, deq_pos_ids, deq_sent_ids)
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1).to(logits.device)
                    loss = self.ce(logits, labels)
                    # reconstruction contraints
                    if len(encoder_layer_input) != 0 and len(encoder_layer_input) == len(decoder_layer_output):
                        for i in range(len(encoder_layer_input)):
                            if lambda1[i] != 0:
                                loss += lambda1[i] * F.mse_loss(encoder_layer_input[i], decoder_layer_output[i])
                    # independence contraints
                    if self.args.num_heads > 1 and len(rec_layer_hsic) != 0:
                        batch_size = rec_layer_hsic[0].shape[0]
                        label = torch.arange(self.args.num_heads)
                        label = torch.tile(label, [batch_size * self.args.maxlen, 1]).to(self.args.device)
                        for l in range(len(rec_layer_hsic)):
                            if lambda2[l] != 0:
                                loss += lambda2[l] * F.nll_loss(rec_layer_hsic[l].view(batch_size * self.args.maxlen, self.args.num_heads, self.args.num_heads), label)
                    # backward
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    t.set_postfix(loss=loss.item())
                # evaluate model
                if ((epoch+1) % self.eval_interval == 0 or epoch + 1 == self.num_epochs):
                    self.model.eval()
                    t_test, test_AUC = self.evaluate(mode='test')
                    t_valid, valid_AUC = self.evaluate(mode='val')
                    t1 = time.time() - t0
                    T += t1
                    print('Evaluating', end='')
                    for k in self.metric_ks:
                        print(f"epoch: {epoch + 1}, time: {T}, valid (NDCG@{k}: {t_valid[0][k]:.4f}, HR@{k}: {t_valid[1][k]:.4f}, AUC: {valid_AUC}), test (NDCG@{k}: {t_test[0][k]:.4f}, HR@{k}: {t_test[1][k]:.4f}, AUC: {test_AUC})")
                    valid_score = valid_AUC
                    if valid_score >= best_valid_score:
                        best_valid_score = valid_score
                        best_epoch = epoch
                        rec_valid = t_valid
                        rec_test = t_test
                        rec_valid_AUC = valid_AUC
                        rec_test_AUC = test_AUC
                    t0 = time.time()
        for k in self.metric_ks:
            print(f"epoch: {best_epoch}, time: {T}, valid (NDCG@{k}: {rec_valid[0][k]:.4f}, HR@{k}: {rec_valid[1][k]:.4f}, AUC: {rec_valid_AUC}), test (NDCG@{k}: {rec_test[0][k]:.4f}, HR@{k}: {rec_test[1][k]:.4f}, AUC: {rec_test_AUC})")
        return best_epoch, rec_valid, rec_valid_AUC, rec_test, rec_test_AUC

    def _get_datainfo_loader(self, args):
        '''
        get dataset's information and convert to torch.utils.data.DataLoader
        '''
        dataset = data_partition(args.dataset)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset
        generate = True
        # generate dataset from https://github.com/FeiSun/BERT4Rec/blob/master/gen_data.py
        for u in user_train:
            if u in user_valid:
                user_train[u].extend(user_valid[u])
        # train dataset
        train_dataset = BertTrainDataset(
            user_train=user_train,
            user_val=user_valid,
            user_test=user_test,
            usernum=usernum,
            itemnum=itemnum,
            maxlen=args.maxlen,
            negative_sampler=RandomSampler(
                user_train,
                user_valid,
                user_test,
                usernum,
                itemnum,
                args.train_negative_sample_size
            ),
            mask_prob=args.mask_prob,
            seed=args.dataset_random_seed,
            generate=generate,
            dupe_factor=args.dupe_factor,
            prop_sliding_window=args.prop_sliding_window
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
        popular_sampler = PopularSampler(
            user_train,
            user_valid,
            user_test,
            usernum,
            itemnum,
            args.eval_negative_sample_size
        )
        # validation dataset
        val_dataset = BertEvalDataset(
            user_train=user_train,
            user_val=user_valid,
            user_test=user_test,
            usernum=usernum,
            itemnum=itemnum,
            maxlen=args.maxlen,
            negative_sampler=popular_sampler,
            mode='val',
        )
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, num_workers=4, shuffle=False)
        # test dataset
        test_dataset = BertEvalDataset(
            user_train=user_train,
            user_val=user_valid,
            user_test=user_test,
            usernum=usernum,
            itemnum=itemnum,
            maxlen=args.maxlen,
            negative_sampler=popular_sampler,
            mode='test',
        )
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=4, shuffle=False)
        return usernum, itemnum, train_loader, val_loader, test_loader
