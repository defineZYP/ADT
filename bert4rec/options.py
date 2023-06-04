import argparse
from utils import set_template

parser = argparse.ArgumentParser()

# base
parser.add_argument('--dataset', help='dataset name', type=str, default='ml-1m') # dataset
parser.add_argument('--min_rating', help='keep ratings greater than equal to this value', type=int, default=4)
parser.add_argument('--min_uc', help='keep users with more than min_uc ratings', type=int, default=5)
parser.add_argument('--min_sc', help='keep items with more than min_sc ratings', type=int, default=0)
parser.add_argument('--dataset_random_seed', help='random seed of split dataset, used for mask data', type=int, default=23)
parser.add_argument('--eval_set_size', help='maximum number of users when evaluation, negative value means all users needed to be evaluated', type=int, default=-1)
parser.add_argument('--topk', help='use which lambda set', type=int, default=-1)


# loader
parser.add_argument('--dataloader_random_seed', help='dataloader random seed', type=float, default=23)
parser.add_argument('--batch_size', help='batch size', type=int, default=256)
parser.add_argument('--eval_batch_size', help='batch size when evaluation', type=int, default=512)

# negative sample
parser.add_argument('--train_negative_sample_size', help='negative sample size when training', type=int, default=100)
parser.add_argument('--eval_negative_sample_size',  help='negative sample size when evaluation', type=int, default=100)

# trainer
parser.add_argument('--device', help='device', type=str, default='cuda')
parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
parser.add_argument('--weight_decay', help='weight decay', type=float, default=0.001)
parser.add_argument('--num_epochs', help='num epochs', type=int, default=100)
parser.add_argument('--clip', help='gradient clip', type=int, default=5)

# evaluate
parser.add_argument('--eval_interval', help='evaluation interval', type=int, default=1)
parser.add_argument('--metric_ks', help='@k metrics, for example, [5, 10] means we need to evaluate HR@5 and HR@10', nargs='+', type=int, default=[5, 10])

# model and dataset, reimplemented from https://github.com/FeiSun/BERT4Rec
parser.add_argument('--dupe_factor', help='number of masked data from each user', type=int, default=10)
parser.add_argument('--prop_sliding_window', help='the masked part of the user-item sequence', type=float, default=0.1)
parser.add_argument('--type_vocab_size', help='number of vocab size', type=int, default=2)
parser.add_argument('--initializer_range', help='initializer range', type=float, default=0.02)
parser.add_argument('--maxlen', help='max length of user-item sequence', type=int, default=200)
parser.add_argument('--hidden_units', help='hidden units of transformer input and output', type=int, default=256)
parser.add_argument('--inner_units', help='hidden units in feed forward networks', type=int, default=1024)
parser.add_argument('--num_layers', help='number of layers', type=int, default=2)
parser.add_argument('--num_heads', help='number of attention heads', type=int, default=2)
parser.add_argument('--dropout', help='dropout rate in fwd networks', type=float, default=0.2)
parser.add_argument('--attention_dropout', help='dropout rate in attention modules', type=float, default=0.2)
parser.add_argument('--mask_prob', help='mask probability', type=float, default=0.2)

# templates
parser.add_argument('--template', help='whether to use template', type=bool, default=True)
parser.add_argument('--experiment_dir', help='output dir', type=str, default='./experiment/')
parser.add_argument('--experiment_description', help='description', type=str, default='test')

# evolution algo
parser.add_argument('--warmup_epochs', help='number of epochs when warmup training', default=200, type=int)
parser.add_argument('--search_epochs', help='number of epochs when searching best lambdas', default=500, type=int)
parser.add_argument('--population_num', help='population number', type=int, default=100)
parser.add_argument('--select_num', help='select topk candidates', type=int, default=50)
parser.add_argument('--m_prob', help='probability of crossover and mutation during evolution process', type=float, default=0.1)
parser.add_argument('--crossover_num', help='crossover number', type=int, default=25)
parser.add_argument('--mutation_num', help='mutation number', type=int, default=25)
parser.add_argument('--seed', help='random seed during evolution', type=int, default=2022)
parser.add_argument('--scale_factor', help='scale factor', type=float, default=0.5)
parser.add_argument('--scale_decay_rate', help='scale decay rate', type=float, default=0.5)

args = parser.parse_args()
if args.template:
    args = set_template(args)

print(args)
