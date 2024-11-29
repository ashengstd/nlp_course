import json
import random

import torch

BUFSIZE = 4096000


def ListsToTensor(xs, tknizer=None):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = tknizer.token2idx(x) + [tknizer.padding_idx] * (max_len - len(x)) if tknizer is not None else x + [0] * (max_len - len(x))
        ys.append(y)
    return ys


def batchify(data, tknizer):
    truth, inp, msk = [], [], []
    for x in data:
        inp.append(x[:-1])
        truth.append(x[1:])
        msk.append([1 for i in range(len(x) - 1)])

    truth = torch.LongTensor(ListsToTensor(truth, tknizer)).t_().contiguous()
    inp = torch.LongTensor(ListsToTensor(inp, tknizer)).t().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return truth, inp, msk


def s2t(strs, tknizer):
    inp, msk = [], []
    for x in strs:
        inp.append([w for w in x])
        msk.append([1 for i in range(len(x))])

    inp = torch.LongTensor(ListsToTensor(inp, tknizer)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return inp, msk


def split_into_chunks(lst, chunk_size):
    chunk_size = max(1, chunk_size)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


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
        sents = split_into_chunks(line, max_len)
        if len(sents[-1]) < min_len:
            sents = sents[:-1]
        data.extend(sents)
    return data


def parse_lines_toutiao(lines, max_len, min_len):
    data = []
    for line in lines:
        # 以 '_!_' 分割每行数据
        parts = line.strip().split("_!_")
        if len(parts) < 5:
            continue  # 跳过不完整的行

        category_name = parts[2]
        news_title = parts[3]

        # 创建一个字典来存储每条新闻的信息
        line = f"标题：{news_title}\n 类别：{category_name}"
        line = list(line)
        sents = split_into_chunks(line, max_len)
        if len(sents[-1]) < min_len:  # the last one is too short
            sents = sents[:-1]
        data.extend(sents)
    return data


class DataLoader:
    def __init__(self, tknizer, filename, batch_size, max_len, min_len, parse_func):
        self.batch_size = batch_size
        self.tknizer = tknizer
        self.max_len = max_len
        self.min_len = min_len
        self.filename = filename
        self.epoch_id = 0
        self.parse_func = parse_func

    def __iter__(self):
        while True:
            with open(self.filename, encoding="utf8") as stream:
                lines = stream.readlines(BUFSIZE)

                if not lines:
                    self.epoch_id += 1
                    continue

                data = self.parse_func(lines[:-1], self.max_len, self.min_len)
                random.shuffle(data)

                idx = 0
                while idx < len(data):
                    yield batchify(data[idx : idx + self.batch_size], self.tknizer)
                    idx += self.batch_size
