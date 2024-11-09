import json
import random

import torch

BUFSIZE = 4096000


def ListsToTensor(xs, tknizer=None):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        if tknizer is not None:
            y = tknizer.token2idx(x) + [tknizer.padding_idx] * (max_len - len(x))
        else:
            y = x + [0] * (max_len - len(x))
        ys.append(y)
    return ys


def batchify(data, tknizer):
    def prepare_batch(data, offset):
        return [x[offset:] for x in data]

    inp = prepare_batch(data, 0)
    truth = prepare_batch(data, 1)
    msk = [[1] * (len(x) - 1) for x in data]

    truth = torch.LongTensor(ListsToTensor(truth, tknizer)).t_().contiguous()
    inp = torch.LongTensor(ListsToTensor(inp, tknizer)).t().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return truth, inp, msk


def s2t(strs, tknizer):
    inp = [[w for w in x] for x in strs]
    msk = [[1] * len(x) for x in strs]

    inp = torch.LongTensor(ListsToTensor(inp, tknizer)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return inp, msk


def chunks(l, n):
    n = max(1, n)
    return [l[i : i + n] for i in range(0, len(l), n)]


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
        sents = chunks(line, max_len)
        if len(sents[-1]) < min_len:
            sents = sents[:-1]
        data.extend(sents)
    return data


class DataLoader:
    def __init__(self, tknizer, filename, batch_size, max_len, min_len):
        self.batch_size = batch_size
        self.tknizer = tknizer
        self.max_len = max_len
        self.min_len = min_len
        self.filename = filename
        self.epoch_id = 0

    def __iter__(self):
        while True:
            with open(self.filename, encoding="utf8") as stream:
                lines = stream.readlines(BUFSIZE)
                if not lines:
                    self.epoch_id += 1
                    continue

                data = parse_lines(lines[:-1], self.max_len, self.min_len)
                random.shuffle(data)

                for idx in range(0, len(data), self.batch_size):
                    yield batchify(data[idx : idx + self.batch_size], self.tknizer)
