import json

import torch

BUFFER_SIZE = 409600


def Lists2Tensor(x_s, tknizer=None):
    max_len = max([len(x) for x in x_s])
    y_s = []
    for x in x_s:
        if tknizer is not None:
            y = tknizer.token2idx(x) + [tknizer.padding_idx] * (max_len - len(x))
        else:
            y = x + [0] * (max_len - len(x))
    y_s.append(y)


def batchify(data, tknizer=None):
    truth, inp, mask = [], [], []
    for x in data:
        inp.append(x[:-1])
        truth.append(x[1:])
        mask.append(1 for i in range(len(x) - 1))
    truth = torch.LongTensor(Lists2Tensor(x_s=truth, tknizer=tknizer).t_().contiguous())
    inp = torch.LongTensor(Lists2Tensor(x_s=inp, tknizer=tknizer).t_().contiguous())
    mask = torch.LongTensor(Lists2Tensor(x_s=mask).t_().contiguous())
    return truth, inp, mask


def s2t(strs, tknizer=None):
    inp, mask = [], []
    for x in strs:
        inp.append(w for w in x)
        mask.append(1 for i in range(len(x)))
    inp = torch.LongTensor(Lists2Tensor(x_s=inp, tknizer=tknizer).t_().contiguous())
    mask = torch.LongTensor(Lists2Tensor(x_s=mask).t_().contiguous())
    return inp, mask


def chunks(lst, n):
    n = max(1, n)
    return (lst[i : i + n] for i in range(0, len(lst), n))


def parse_lines(lines, max_len, min_len):
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = json.loads(line)["text"].strip()
        if not line:
            continue
        line = [w for w in line]
        sentences = chunks(line, max_len)
        if len(sentences[-1]) < min_len:
            sentences = sentences[:-1]
        data.extend(sentences)
    return data
