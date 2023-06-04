import os
import tqdm
import json
import torch
import random

import numpy as np

from datetime import date
from collections import defaultdict, Counter

def setup_train(args):
    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)
    return export_root

def create_experiment_export_folder(args):
    experiment_dir, experiment_description = args.experiment_dir, args.experiment_description
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description)
    os.mkdir(experiment_path)
    print('Folder created: ' + os.path.abspath(experiment_path))
    return experiment_path

def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path


def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx


def load_weights(model, path):
    pass


def save_test_result(export_root, result):
    filepath = Path(export_root).joinpath('test_result.txt')
    with filepath.open('w') as f:
        json.dump(result, f, indent=2)


def export_experiments_config_as_json(args, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=2)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_pretrained_weights(model, path):
    chk_dict = torch.load(os.path.abspath(path))
    model_state_dict = chk_dict[STATE_DICT_KEY] if STATE_DICT_KEY in chk_dict else chk_dict['state_dict']
    model.load_state_dict(model_state_dict)


def setup_to_resume(args, model, optimizer):
    chk_dict = torch.load(os.path.join(os.path.abspath(args.resume_training), 'models/checkpoint-recent.pth'))
    model.load_state_dict(chk_dict[STATE_DICT_KEY])
    optimizer.load_state_dict(chk_dict[OPTIMIZER_STATE_DICT_KEY])


def create_optimizer(model, args):
    if args.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

def set_template(args, template_folder='./templates'):
    filename = f"{template_folder}/{args.dataset}.json"
    with open(filename, 'r') as f:
        arg_dic = json.load(f)
    for k in arg_dic:
        setattr(args, k, arg_dic[k])
    return args

def evaluate_loader(model, loader, args, mode='val', ks=[5, 10]):
    NDCG = {k: 0.0 for k in ks}
    HT = {k: 0.0 for k in ks}
    valid_user = 0.0
    ranks_all = []
    with tqdm.tqdm(loader) as t:
        t.set_description(f"Evaluate {mode}: ")
        for batch, label in t:
            u, seq, item_idx = batch
            seq_pos_ids = torch.LongTensor(np.tile(np.array(range(seq.shape[1])), [seq.shape[0], 1])).to(args.device)
            seq_sent_ids = torch.zeros(seq.shape, dtype=int).to(args.device)
            predictions = -model.predict(u, seq, seq_pos_ids, seq_sent_ids, item_idx)
            # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            # predictions = -model.predict(u, seq, item_idx)
            # for given metric, calculate it 
            # batch_size * 1
            rank = predictions.argsort(dim=1).argsort(dim=1)[:, 0]
            ranks_all.extend(rank.cpu().detach().numpy())
            valid_user += u.shape[0]
            for k in ks:
                rank_matrix = torch.where(rank < k)
                hit_matrix = rank[rank_matrix]
                ndcg_matrix = 1 / torch.log2(hit_matrix + 2)
                HT[k] += hit_matrix.shape[0]
                NDCG[k] += ndcg_matrix.sum()
            # candidates_size = item_idx.shape[1]
            # AUC = 
    for k in ks:
        HT[k] = HT[k] / valid_user
        NDCG[k] = NDCG[k].item() / valid_user
    # AUC = roc_auc_score(label_all, logits_all)
    ranks_all = np.array(ranks_all) + 1
    candidates_size = 1 + item_idx.shape[1]
    AUC = np.mean((candidates_size - ranks_all) / (candidates_size - 1))
    return (NDCG, HT), AUC

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_lambda(dataset, tp=-1):
    '''
    according to dataset get lambda_1(reconstruction) and lambda_2(independence)
    '''
    if dataset == 'ml-1m':
        return [0.001033064113633401, 5.277219708128945e-06], [0.000899362502660037, 0.000706016178174784]
    if dataset == 'beauty' or dataset == "Beauty":
        return [1.4616741512829565e-05, 0.001839446918736823], [0.00037889972403308536, 0.0009180599125696732]
    if dataset == 'steam':
        return [0.0003957887657578212, 6.360759018525728e-05], [0.0010088509057684678, 0.0008035241708960854]
    if dataset == 'ml-20m':
        return [0.005435293808249262, 0.0019764407654292064], [0.0007068258408279514, 0.0013811031763964325]
