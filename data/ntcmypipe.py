from fastNLP.io import ConllLoader, Loader
from fastNLP.io.loader.conll import _read_conll
from fastNLP.io.pipe.utils import iob2, iob2bioes
from fastNLP import DataSet, Instance
from fastNLP.io import Pipe
from tokenizers.implementations import BertWordPieceTokenizer
from fastNLP.core.metrics import _bio_tag_to_spans
from fastNLP.io import DataBundle
import numpy as np
from itertools import chain
from fastNLP import Const
from functools import cmp_to_key
import json
from copy import deepcopy
from tqdm import tqdm
import torch
UNK = "[UNK]"
unk_token_map = {"“": '"',
                 "”": '"',
                 "‘": "'",
                 "’": "'"}

class BartNERPipe(Pipe):
    def __init__(self, tokenizer='bart-base-chinese', dataset_name='ntcdata', target_type='word'):
        """

        :param tokenizer:
        :param dataset_name:
        :param target_type:
            支持word: 生成word在词表中的位置;
            bpe: 生成所有的bpe
            span: 每一段按照start end生成
            span_bpe: 每一段都是start的所有bpe，end的所有bpe
        """
        super().__init__()
        self.tokenizer = BertWordPieceTokenizer("vocab.txt")

        assert target_type in ('word', 'bpe', 'span')

        if dataset_name == 'ntcdata':
            self.mapping = {
                'nolack': '[nolack]',
                'stack': '[stack]',
                'new_branch': '[new_branch]',
                'postposition': '[postposition]',
                'influx_chunks': '[influx_chunks]'
            }  # 记录的是原始tag与转换后的tag的str的匹配关系
        else:
            raise
        cur_num_tokens = self.tokenizer.get_vocab_size()
        self.num_token_in_orig_tokenizer = cur_num_tokens
        self.target_type = target_type
    def add_tags_to_special_tokens(self, data_bundle):

        self.mapping2id = {}  # 给定转换后的tag，输出的是在tokenizer中的id，用来初始化表示
        self.mapping2targetid = {}  # 给定原始tag，输出对应的数字

        for key, value in self.mapping.items():
            key_id = self.tokenizer.token_to_id(value)
            self.mapping2id[value] = key_id #
            self.mapping2targetid[key] = len(self.mapping2targetid)


    def process(self, data_bundle):
        """
        支持的DataSet的field为

            entities: List[List[str]], 每个元素是entity，非连续的拼到一起了
            entity_tags: 与上面一样长，是每个entity的tag
            raw_words: List[str]词语
            entity_spans： List[List[int]]记录的是上面entity的start和end，这里的长度一定是偶数，是start,end的pair, end是开区间

        :param data_bundle:
        :return:
        """
        self.add_tags_to_special_tokens(data_bundle)

        # 转换tag
        target_shift = len(self.mapping) + 2  # 是由于第一位是sos，紧接着是eos, 然后是

        def prepare_target(ins):
            raw_words = ins['raw_words']
            word_bpes = [self.tokenizer.token_to_id('[CLS]')]
            first = [0]  # 用来取每个word第一个bpe

            for id, word in enumerate(raw_words):
                bpes = self.tokenizer.token_to_id(word)
                if bpes is None:
                    bpes = self.tokenizer.token_to_id('[UNK]')
                word_bpes.append(bpes)
                first.append(id + 1)
            word_bpes.append(self.tokenizer.token_to_id('[SEP]'))
            first.append(len(first))
            assert len(first) == len(raw_words) + 2 == len(word_bpes)
            entity_spans = ins['entity_spans'] # [(s1, e1, s2, e2), ()]
            entity_tags = ins['entity_tags']# [tag1, tag2...]
            entities = ins['entities']# [[ent1, ent2,], [ent1, ent2]]
            if entity_spans[0][-1] != 0:
                print(word_bpes[entity_spans[0][0]: entity_spans[0][1]])
                print([self.tokenizer.token_to_id(i) if self.tokenizer.token_to_id(i) else self.tokenizer.token_to_id('[UNK]') for i in entities[0][0]])
                assert word_bpes[entity_spans[0][0]: entity_spans[0][1]] == [
                    self.tokenizer.token_to_id(i) if self.tokenizer.token_to_id(i) else self.tokenizer.token_to_id('[UNK]') for i in entities[0][0]]
            target = [0]  # 特殊的sos
            pairs = []

            entity, tag = entity_spans[0], entity_tags[0]
            cur_pair = []
            if entity[1] != 0:
                cur_pair = [target_shift + pos for pos in range(entity[0], entity[1])]

            cur_pair.append(self.mapping2targetid[tag] + 2)  # 加2是由于有shift
            pairs.append([p for p in cur_pair])
            target.extend(list(chain(*pairs)))
            target.append(1)  # 特殊的eos
            if len(word_bpes) > 511:
                word_bpes = word_bpes[:512]
            assert len(word_bpes) <= 512

            if len(first) > 511:
                word_bpes = first[:512]
            assert len(first) <= 512

            dict = {'tgt_tokens': target, 'target_span': pairs, 'src_tokens': word_bpes,
                    'first': first, 'entities': entities}
            return dict

        data_bundle.apply_more(prepare_target, use_tqdm=True, tqdm_desc='pre. tgt.')

        data_bundle.set_ignore_type('target_span', 'entities')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.token_to_id("[PAD]"))

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities')


        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据

        if "ntcdata" in paths['train']:
            data_bundle = NtcLoader(demo=demo).load(paths)
        else:
            raise
        data_bundle = self.process(data_bundle)
        return data_bundle

class NtcLoader(ConllLoader):
    def __init__(self, demo=False):
        self.demo = demo

    def count_sent(self, text):
        count_b = 0
        for w in text:
            if w == '[MASK]':
                count_b += 1
        return count_b

    def find_mask_abspos(self, text, pos):
        pos_count = 1
        for idx, w in enumerate(text):
            if w == '[MASK]' and pos_count == pos:
                return idx
            if w == '[MASK]':
                pos_count += 1
        return idx

    def changeMask(self, x):
        if x == '<b>':
            return '[MASK]'
        else:
            return x

    def _load(self, paths):
        ds = DataSet()

        def changeMask(x):
            if x == '[MASK]':
                return 'Ё'
            else:
                return x

        f = open(paths, 'r')
        for i, ntc_line in enumerate(f.readlines()):
            line_ntc = json.loads(ntc_line)
            text_ori = line_ntc['text']
            text = [self.changeMask(w) for w in text_ori]

            answers = line_ntc['answers']
            ntc_lack_c = dict()
            ntc_lack_p = dict()
            entity_tags = dict()
            if len(answers) != 0:
                for an in answers:
                    answer = an['answer']
                    answer_start = an['answer_start']
                    answer_end = an['answer_end']
                    position = an['position']
                    tag = an['type'].strip()
                    entity_tags[position] = tag
                    ntc_lack_c[position] = answer
                    ntc_lack_p[position] = [answer_start, answer_end + 1]
            else:
                entity_tags = None
                ntc_lack_c = None
                ntc_lack_p = None

            sentCount = self.count_sent(text)
            for i in range(1, sentCount + 1):
                entity_span = []
                entity = []
                abs_pos = self.find_mask_abspos(text, i)
                if abs_pos >= 507:
                    continue

                text2 = ''.join([changeMask(w) for w in text])
                text1 = text2[:abs_pos].replace("Ё", "") + "[MASK]" + text2[abs_pos:].replace("Ё", "")
                text1 = text1.split("[MASK]")
                raw_words = [w if w not in unk_token_map else unk_token_map[w] for w in text1[0]] + ['[MASK]'] + \
                            [w if w not in unk_token_map else unk_token_map[w] for w in text1[1]]
                if len(raw_words) >= 510:
                    continue
                mask_pos = raw_words.index('[MASK]')
                if ntc_lack_c is not None and i in ntc_lack_c:
                    target = [[w if w not in unk_token_map else unk_token_map[w] for w in ntc_lack_c[i]]]
                    entity.append(target)
                    entity_tag = [entity_tags[i].lower()]
                    if ntc_lack_p[i][0] < mask_pos:
                        entity_span.append(ntc_lack_p[i])
                    else:
                        start = ntc_lack_p[i][0] + 1
                        end = ntc_lack_p[i][1] + 1
                        entity_span.append([start, end])

                    if entity_span[0][-1] >= 510:
                        continue
                else:
                    target = ['nolack']
                    entity.append(target)
                    entity_span.append([0, 0])
                    entity_tag = target

                ds.append(Instance(raw_words=raw_words, entities=entity, entity_tags=entity_tag,
                               entity_spans=entity_span, raw_target=target))

            if self.demo and len(ds) > 3:
                break
        if len(ds) == 0:
            raise RuntimeError("No data found {}.".format(paths))
        return ds


def cmp(v1, v2):
    v1 = v1[-1]
    v2 = v2[-1]
    if v1[0] == v2[0]:
        return v1[-1] - v2[-1]
    return v1[0] - v2[0]


if __name__ == '__main__':
    data_bundle = Conll2003NERLoader(demo=False).load('conll2003')
    BartNERPipe(target_type='word', dataset_name='conll2003').process(data_bundle)
