import os
import json
import gc
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse

import torch.nn.functional as F
from tqdm import trange

from model import SASRecADT
from utils import *

# torch.autograd.set_detect_anomaly(True)

def str2bool(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--clip', default=5.0, type=float)
    parser.add_argument('--warmup_steps_rate', default=0.1, type=float)
    parser.add_argument('--sample_size', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--is_save', default=False, type=str2bool)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--eval_interval', default=20, type=int)
    parser.add_argument('--eval_batch_size', default=512, type=int)
    parser.add_argument('--eval_set', default=-1, type=int)
    parser.add_argument('--ind_loss', default='dcov', type=str)
    parser.add_argument('--vis', default=False, type=str2bool)
    parser.add_argument('--topk', default=-1, type=int)
    # parser.add_argument('--local_rank', default=-1, type=int)

    args = parser.parse_args()
    args = set_template(args)

    print(args)
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()
    return args

def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

def main(args_str=None):
    args = parse_args()
    # local_rank = args.local_rank
    set_rng_seed(23)

    lambdas1, lambdas2 = get_lambdas(args.dataset, args.topk)

    args.num_layers = len(lambdas1)
    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    negative_sampler = PopularSampler(user_train, user_valid, user_test, usernum, itemnum, args.sample_size)
    num_batch = len(user_train) // args.batch_size
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    
    # sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    warp_dataset = WarpDataset(user_train, usernum, itemnum, args.maxlen)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(warp_dataset)
    dataloader = DataLoader(warp_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataset = EvalDataset(user_train, user_valid, user_test, usernum, itemnum, args.maxlen, negative_sampler, mode='val', eval_set=args.eval_set)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, num_workers=4, shuffle=False)
    test_dataset = EvalDataset(user_train, user_valid, user_test, usernum, itemnum, args.maxlen, negative_sampler, mode='test', eval_set=args.eval_set)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=4, shuffle=False)
    model = SASRecADT(usernum, itemnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            # print('failed loading state_dicts, pls check file path: ', end="")
            # print(args.state_dict_path)
            # print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()

    # model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    model.train() # enable model training
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
   
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    # warmup_steps = int(num_batch * args.num_epochs * args.warmup_steps_rate)
    T = 0.0
    t0 = time.time()

    best_valid_score = 0.0
    best_epoch = 0
    rec_valid = []
    rec_test = []
    rec_AUC_valid = 0.0
    rec_AUC_test = 0.0
    losses = []
    lrs = []
    ks = [5, 10]
    times = []
    
    for epoch in range(args.num_epochs):
        # calculate_start = time.time()
        if args.inference_only: break # just to decrease identition
        with tqdm.tqdm(dataloader) as t: # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            for batch, _ in t:
                t.set_description(f"Warmup: epoch {epoch + 1} / {args.num_epochs} ")
                u, seq, dec, pos, neg = batch # tuples to ndarray
                u, seq, dec, pos, neg = np.array(u), np.array(seq), np.array(dec), np.array(pos), np.array(neg)
                pos_logits, neg_logits, encoder_layer_input, decoder_layer_output, rec_layer_ind = model(u, seq, dec, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
                # print(pos_logits, neg_logits)
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                # recon
                if len(encoder_layer_input) != 0 and len(encoder_layer_input) == len(decoder_layer_output):
                    # MSE loss calculate the reconstruction loss
                    for i in range(len(encoder_layer_input)):
                        loss += lambdas1[i] * F.mse_loss(encoder_layer_input[i], decoder_layer_output[i])
                
                if args.num_heads > 1:
                    # ind loss
                    # generate label
                    batch_size = rec_layer_ind[0].shape[0]
                    label = torch.arange(args.num_heads)
                    label = torch.tile(label, [batch_size * args.maxlen, 1]).to(args.device)
                    # calculate loss
                    for l in range(len(rec_layer_ind)):
                        # rec_layer_ind[i] shape: [batch_size, maxlen, num_head, num_head]
                        loss += lambdas2[i] * F.nll_loss(rec_layer_ind[l].view(batch_size * args.maxlen, args.num_heads, args.num_heads), label)
                for param in model.item_emb.parameters(): loss += args.weight_decay * torch.norm(param)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                adam_optimizer.step()
                t.set_postfix(loss=loss.item())
                losses.append(loss.item())
                lrs.append(adam_optimizer.param_groups[0]['lr'])

        if (((epoch+1) % args.eval_interval == 0) or (epoch + 1 == args.num_epochs)):
            print(f"{epoch}/{args.num_epochs}")
            model.eval()
            good_start = time.time()
            t_test, AUC_test = evaluate_loader(model, test_loader, args, 'test', ks)
            t_valid, AUC_valid = evaluate_loader(model, val_loader, args, 'val', ks)
            good_end = time.time()
            print(f"{good_start - good_end} s inference")
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            for k in ks:
                print(f"epoch: {epoch + 1}, time: {T}, valid (NDCG@{k}: {t_valid[0][k]:.4f}, HR@{k}: {t_valid[1][k]:.4f}, AUC: {AUC_valid}), test (NDCG@{k}: {t_test[0][k]:.4f}, HR@{k}: {t_test[1][k]:.4f}, AUC: {AUC_test})")
                # print('epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                #         % (epoch, T, k, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            valid_score = 0.0
            if AUC_valid >= best_valid_score:
                best_valid_score = AUC_valid
                best_epoch = epoch
                rec_valid = t_valid
                rec_test = t_test
                rec_AUC_valid = AUC_valid
                rec_AUC_test = AUC_test

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            if args.is_save:
                # print("?????")
                folder = f"/root/autodl-tmp/sasrec/{args.dataset}_{args.topk}/checkpoints"
                os.makedirs(folder, exist_ok=True)
                fname = os.path.join(folder, f"SASRec.epoch={epoch}.lr={args.lr}.layer={args.num_layers}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth")
                torch.save(model.state_dict(), fname)
            model.train()
    
        if epoch == args.num_epochs and args.is_save:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    for k in ks:
        print(f"epoch: {best_epoch}, time: {T}, valid (NDCG@{k}: {rec_valid[0][k]:.4f}, HR@{k}: {rec_valid[1][k]:.4f}, AUC: {rec_AUC_valid}), test (NDCG@{k}: {rec_test[0][k]:.4f}, HR@{k}: {rec_test[1][k]:.4f}, AUC: {rec_AUC_test})")

    f.close()
    # sampler.close()
    return best_epoch, rec_valid, rec_valid, rec_test, rec_test
    # print("Done")

if __name__ == '__main__':
    main()
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
    