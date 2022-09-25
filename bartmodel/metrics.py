
from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
import numpy as np
import torch


class Seq2SeqSpanMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels, target_type='bpe'):
        super(Seq2SeqSpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels + 2  # +2是由于有前面有两个特殊符号，sos和eos

        self.tptrue = 0
        self.slacknum = 0
        self.placknum = 0
        self.spantptrue = 0
        self.spanslacknum = 0
        self.spanplacknum = 0
        self.correct = 0
        self.span_correct = 0
        self.em = 0
        self.total = 0
        self.target_type = target_type  # 如果是span的话，必须是偶数的span，否则是非法的

    def evaluate(self, target_span, pred, tgt_tokens):
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # 去掉<s>
        tgt_tokens = tgt_tokens[:, 1:] # 去掉<s>
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1) # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1) # bsz
        target_seq_len = (target_seq_len - 2).tolist()
        for i, (ts, ps) in enumerate(zip(target_span, pred.tolist())):
            em = 0
            ps = ps[:pred_seq_len[i]]
            if pred_seq_len[i] == target_seq_len[i]:
                em = int(tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item()==target_seq_len[i])
            self.em += em
            pairs = []
            cur_pair = []
            if len(ps):
                for j in ps:
                    if j < self.word_start_index:
                        if len(cur_pair) > 0:
                            if all([cur_pair[i]<cur_pair[i+1] for i in range(len(cur_pair)-1)]):
                                pairs.append(tuple(cur_pair + [j]))
                        cur_pair = []
                    else:
                        cur_pair.append(j)
            if not pairs:
                pairs.append([2])

            flag = 1
            for i, j in zip(pairs, ts):
                for ii, jj in zip(i, j):
                    if not (ii==jj):
                        flag = 0
            if flag:
                self.correct += 1

            # 生成span
            flag = 1
            for i, j in zip(pairs, ts):
                for ii, jj in zip(i[-1:], j[-1:]):
                    if not (ii == jj):
                        flag = 0
            if flag:
                self.span_correct += 1

            slacknum, placknum, tptrue = _compute_tp_fn_fp(pairs, ts)
            self.slacknum += slacknum
            self.placknum += placknum
            self.tptrue += tptrue

            spanStrue, spanPtrue, spanTPtrue = span_compute_tp_fn_fp(pairs, ts)
            self.spantptrue += spanTPtrue
            self.spanslacknum += spanStrue
            self.spanplacknum += spanPtrue


    def get_metric(self, reset=True):
        res = {}
        pre = self.tptrue / (self.placknum + 1e-8)
        rec = self.tptrue / (self.slacknum + 1e-8)
        f = 2 *pre *rec / (pre + rec + 1e-8)
        spanp = self.spantptrue / (self.spanplacknum + 1e-8)
        spanr = self.spantptrue / (self.spanslacknum + 1e-8)
        spanf = 2 * pre * rec / (spanp + spanr + 1e-8)
        res['f'] = round(f, 4)
        res['r'] = round(rec, 4)
        res['p'] = round(pre, 4)
        res['em'] = round(self.em / self.total, 4)
        res['acc'] = round(self.correct / self.total, 4)
        res['spanf'] = round(spanf, 4)
        res['spanr'] = round(spanr, 4)
        res['spanp'] = round(spanp, 4)
        res['spanacc'] = round(self.span_correct / self.total, 4)
        if reset:
            self.total = 0
            self.tptrue = 0
            self.placknum = 0
            self.slacknum = 0
            self.em = 0
            self.correct = 0
            self.span_correct = 0
            self.spantptrue = 0
            self.spanplacknum = 0
            self.spanslacknum = 0

        return res


def _compute_tp_fn_fp(ps, ts):
    ps = ps.copy()
    ts = ts.tolist()
    slacknum = 0
    placknum = 0
    tptrue = 0

    for key in ts:
        if key != [2]:
            slacknum += 1
    for key in ps:
        if key != [2]:
            placknum += 1

    for i, j in zip(ts, ps):
        if i != [2]:
            flag = 1
            for ii, jj in zip(i, j):
                if ii != jj:
                    flag = 0
            if flag:
                tptrue += 1

    return slacknum, placknum, tptrue


def span_compute_tp_fn_fp(ps, ts):
    ps = ps.copy()
    ts = ts.tolist()
    slacknum = 0
    placknum = 0
    tptrue = 0

    for key in ts:
        if key != [2]:
            slacknum += 1
    for key in ps:
        if key != [2]:
            placknum += 1

    for i, j in zip(ts, ps):
        if i != [2]:
            flag = 1
            for ii, jj in zip(i[-1], j[-1]):
                if ii != jj:
                    flag = 0
            if flag:
                tptrue += 1

    return slacknum, placknum, tptrue
