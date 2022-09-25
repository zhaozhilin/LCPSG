import torch
from fastNLP import cache_results
from data.ntc_pipe import BartNERPipe
from fastNLP import SequentialSampler, SortedSampler

from fastNLP import DataSetIter
from fastNLP.core.utils import _move_dict_value_to_device
from tqdm import tqdm
import json


dataset_name = 'ntcdata'
model_path = 'save_models/best_SequenceGeneratorModel_f_2022-08-29-10-53-05-324879'  # you can set args.save_model=1 in train.py
bart_name = 'bart-base-chinese'
target_type = 'word'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#cache_fn = f"caches/data_{bart_name}_{dataset_name}_{target_type}.pt"

demo = False
#@cache_results(cache_fn, _refresh=False)
def get_data():
    pipe = BartNERPipe(tokenizer=bart_name, dataset_name=dataset_name, target_type=target_type)
    if dataset_name == 'ntcdata':
        paths1 = {'train': "./data/ntcdata/train_fs.json",
                 'test': "./data/ntcdata/test_fs.json",
                 'dev': "./data/ntcdata/eval_fs.json"}
        data_bundle = pipe.process_from_file(paths1, demo=demo)
    else:
        data_bundle = pipe.process_from_file(f'../data/{dataset_name}', demo=demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id


data_bundle, tokenizer, mapping2id = get_data()

model = torch.load(model_path)

device = torch.device(device)
model.to(device)
model.eval()

eos_token_id = 0
word_start_index = len(mapping2id) + 2
not_bpe_start = 0

mapping = { '[nolack]': 'nolack',
            '[stack]': 'stack',
            '[new_branch]': 'new_branch',
            '[postposition]': 'postposition',
            '[influx_chunks]': 'influx_chunks'
            }  # 记录的是原始tag与转换后的tag的str的匹配关系

id2label = {k + 2: mapping[v] for k, v in enumerate(mapping2id.keys())}


def get_pairs(ps, word_start_index):
    pairs = []
    cur_pair = []
    for j in ps:
        if j < word_start_index:
            if len(cur_pair) > 0:
                if all([cur_pair[i] < cur_pair[i + 1] for i in range(len(cur_pair) - 1)]):
                    pairs.append(tuple(cur_pair + [j]))
            cur_pair = []
        else:
            cur_pair.append(j)
    if not pairs:
        pairs.append([2])
    return pairs


def get_tgt_tokens(pairs, src_tokens_i, mapping2id, id2label):
    pred_tokens = []
    for pair in pairs:#
        label = pair[-1]
        if label == 2:
            pred_token = 'nolack'
        else:
            if label < word_start_index and label > 1:
                pred_token = id2label[int(label)]
                pred_tokens.append(pred_token)
            idxes = [(p - len(mapping2id) - 2) for p in pair[:-1]]
            start_idx = idxes[0]
            end_idx = idxes[-1]
            pred_token = ''.join([w for w in src_tokens_i[start_idx - 1: end_idx]])
        pred_tokens.append(pred_token)
    return pred_tokens

with open('bart-base-chinese/vocab.txt') as f:
    idx2word = []
    for line in f.readlines():
        word = line.strip()
        idx2word.append(word)

for name in ['test']:
    ds = data_bundle.get_dataset(name)
    ds.set_ignore_type('raw_words', 'raw_target')
    ds.set_target('raw_words', 'raw_target')
    with open(f'preds/{name}50.txt', 'w', encoding='utf-8') as f:
        data_iterator = DataSetIter(ds, batch_size=16, sampler=SequentialSampler())
        for batch_x, batch_y in tqdm(data_iterator, total=len(data_iterator)):
            _move_dict_value_to_device(batch_x, batch_y, device=device)
            src_tokens = batch_x['src_tokens']
            first = batch_x['first']
            src_seq_len = batch_x['src_seq_len']
            tgt_seq_len = batch_x['tgt_seq_len']
            raw_words = batch_y['raw_words']
            raw_targets = batch_y['raw_target']
            pred_y = model.predict(src_tokens=src_tokens, src_seq_len=src_seq_len, first=first)
            pred = pred_y['pred']
            tgt_tokens = batch_y['tgt_tokens']
            pred_eos_index = pred.flip(dims=[1]).eq(eos_token_id).cumsum(dim=1).long()
            pred = pred[:, 1:]  # 去掉</s>
            tgt_tokens = tgt_tokens[:, 1:]
            pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
            pred_seq_len = (pred_seq_len - 2).tolist()
            tgt_seq_len = (tgt_seq_len - 2).tolist()
            for i, ps in enumerate(pred.tolist()):
                em = 0
                ps = ps[:pred_seq_len[i]]
                ts = tgt_tokens[i, :tgt_seq_len[i]]
                pairs, t_pairs = [], []
                if len(ps):
                    pairs = get_pairs(ps, word_start_index)
                if len(ts):
                    t_pairs = get_pairs(ts, word_start_index)
                raw_words_i = raw_words[i]
                src_tokens_i = src_tokens[i, :src_seq_len[i]].tolist()

                target_y = raw_targets[i]
                pred_tokens = get_tgt_tokens(pairs, raw_words_i, mapping2id, id2label)
                target_tokens = get_tgt_tokens(t_pairs, raw_words_i, mapping2id, id2label)
                idex = min(max(len(target_tokens), len(pred_tokens)), 2)

                for i in range(idex):
                    f.write(f'{i} ')
                    if i < len(target_tokens):
                        f.write(f'{target_tokens[i]} ')
                    else:
                        f.write('None ')
                    if i < len(pred_tokens):
                        f.write(f'{pred_tokens[i]}\n')
                    else:
                        f.write('None\n')

                f.write('\n')

print(f"In total, has {not_bpe_start} predictions on the non-word start.")
