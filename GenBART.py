import argparse
import json
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import sys
sys.path.append('../')
import os

import warnings
warnings.filterwarnings('ignore')
from data.ntcmultipipe import BartNERPipe
import fitlog
from fastNLP import Trainer
from bartmodel.metrics import Seq2SeqSpanMetric
from bartmodel.losses import Seq2SeqLoss
from fastNLP import BucketSampler, GradientClipCallback, cache_results

from bartmodel.callbacks import WarmupCallback
from fastNLP.core.sampler import SortedSampler
from bartmodel.callbacks import FitlogCallback

fitlog.debug()
fitlog.set_log_dir('logs')
from bartmodel.bart import BartSeq2SeqModel
from bartmodel.generater import SequenceGeneratorModel
from torch import optim
writer = SummaryWriter("logs")


parser = argparse.ArgumentParser(description='Training Hyperparams')
# data loading params
parser.add_argument('--data_path', type=str, default='ntc_preprocess', help='Path to the preprocessed data')
# network params
parser.add_argument('--bart_name', type=str, default='bart-base-chinese')
parser.add_argument('--decoder_type', type=str, default='avg_feature')
parser.add_argument('--target_type', type=str, default='word')
parser.add_argument('--length_penalty', type=int, default=2)
parser.add_argument('--max_len', type=int, default=150)
parser.add_argument('--max_len_a', type=float, default=0.8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--dataset_name', type=str, default='ntcdata')
parser.add_argument('--demo', type=bool, default=False)
parser.add_argument('--weighted_model', type=bool, default=False)
parser.add_argument('--beam_size', type=int, default=3, help='Beam width')
parser.add_argument('--n_best', type=int, default=1, help='Output the n_best decoded sentence')
# training params
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--n_epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_src_seq_len', type=int, default=512)
parser.add_argument('--max_tgt_seq_len', type=int, default=30)
parser.add_argument('--max_grad_norm', type=float, default=0)
parser.add_argument('--warmup_ratio', type=float, default=0.01)

opt = parser.parse_args()
print(opt)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_data():
    pipe = BartNERPipe(tokenizer=opt.bart_name, dataset_name=opt.dataset_name, target_type=opt.target_type)
    if opt.dataset_name == 'ntcdata':
        paths1 = {'train': "./data/ntcdatalin/train.json",
                 'test': "./data/ntcdatalin/test.json",
                 'dev': "./data/ntcdatalin/test.json"}
        data_bundle = pipe.process_from_file(paths1, demo=opt.demo)
    else:
        data_bundle = pipe.process_from_file(f'../data/{opt.dataset_name}', demo=opt.demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id

if __name__ == '__main__':
   random_seed = 2022
   n_gpu = torch.cuda.device_count()
   set_seed(random_seed)
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   # 加载数据
   print('Loading training and development data..')
   data_bundle, tokenizer, mapping2id = get_data()
   # 定义预测的类型标签
   label_ids = list(mapping2id.values())
   bos_token_id = 0
   eos_token_id = 1
   # 定义模型
   model = BartSeq2SeqModel.build_model(opt.bart_name, tokenizer, label_ids=label_ids, mapping2id=mapping2id,
                                        decoder_type='avg_feature',
                                        use_encoder_mlp=1)
   vocab_size = 21128
   print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))
   model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                                  eos_token_id=eos_token_id,
                                  max_length=opt.max_len, max_len_a=opt.max_len_a, num_beams=opt.beam_size,
                                  do_sample=False,
                                  repetition_penalty=1, length_penalty=opt.length_penalty, pad_token_id=eos_token_id,
                                  restricter=None)
   # Loss and Optimizer

   parameters = []
   params = {'lr': opt.lr, 'weight_decay': 1e-2}
   params['params'] = [param for name, param in model.named_parameters() if
                       not ('bart_encoder' in name or 'bart_decoder' in name)]
   parameters.append(params)

   params = {'lr': opt.lr, 'weight_decay': 1e-2}
   params['params'] = []
   for name, param in model.named_parameters():
       if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
           params['params'].append(param)
   parameters.append(params)

   params = {'lr': opt.lr, 'weight_decay': 0}
   params['params'] = []
   for name, param in model.named_parameters():
       if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
           params['params'].append(param)
   parameters.append(params)

   optimizer = optim.AdamW(parameters)

   callbacks = []
   callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
   callbacks.append(WarmupCallback(warmup=opt.warmup_ratio, schedule='linear'))

   callbacks.append(FitlogCallback(raise_threshold=0.00, eval_begin_epoch=5))
   eval_dataset = data_bundle.get_dataset('test')

   sampler = BucketSampler(seq_len_field_name='src_seq_len')

   metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), target_type='word')

   ds = data_bundle.get_dataset('train')
   ds.concat(data_bundle.get_dataset('dev'))
   data_bundle.delete_dataset('dev')
   save_path = 'save_models_multi_liu/'
   validate_every = 100000
   trainer = Trainer(train_data=ds, model=model, optimizer=optimizer,
                     loss=Seq2SeqLoss(),
                     batch_size=opt.batch_size, sampler=sampler, drop_last=False, update_every=1,
                     num_workers=0, n_epochs=opt.n_epochs, batch_sampler=None,
                     print_every=1 if 'SEARCH_OUTPUT_FP' not in os.environ else 10,
                     dev_data=eval_dataset, metrics=metric, metric_key='f',
                     validate_every=validate_every, save_path=save_path,
                     use_tqdm='SEARCH_OUTPUT_FP' not in os.environ, device=device,
                     callbacks=callbacks, check_code_level=0, test_use_tqdm='SEARCH_OUTPUT_FP' not in os.environ,
                     test_sampler=SortedSampler('src_seq_len'), dev_batch_size=opt.batch_size * 2)

   trainer.train(load_best_model=False)
   directory = save_path
   torch.save(model, os.path.join(directory, f"model_multi_liu_50"))

   print('Terminated')
