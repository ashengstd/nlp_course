import json
import random

import torch

BUFFER_SIZE = 409600


def Lists2Tensor(x_s, tknizer=None):
    max_len = max(len(x) for x in x_s)
    y_s = []
    for x in x_s:
        if tknizer is not None:
            y = tknizer.token2idx(x) + [tknizer.padding_idx] * (max_len - len(x))
        else:
            y = x + [0] * (max_len - len(x))
        y_s.append(y)
    return y_s


def batchify(data, tknizer=None):
    truth, inp, mask = [], [], []
    for x in data:
        inp.append(x[:-1])
        truth.append(x[1:])
        mask.append([1 for i in range(len(x) - 1)])
    truth = torch.LongTensor(Lists2Tensor(x_s=truth, tknizer=tknizer)).t_().contiguous()
    inp = torch.LongTensor(Lists2Tensor(x_s=inp, tknizer=tknizer)).t_().contiguous()
    mask = torch.LongTensor(Lists2Tensor(x_s=mask)).t_().contiguous()
    return truth, inp, mask


def s2t(strs, tknizer=None):
    inp, mask = [], []
    for x in strs:
        inp.append([w for w in x])  # 使用列表推导式
        mask.append([1 for _ in range(len(x))])  # 使用列表推导式
    inp = torch.LongTensor(Lists2Tensor(x_s=inp, tknizer=tknizer)).t_().contiguous()
    mask = torch.LongTensor(Lists2Tensor(x_s=mask)).t_().contiguous()
    return inp, mask


def chunks(lst, n):
    n = max(1, n)
    return [lst[i : i + n] for i in range(0, len(lst), n)]


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


class DataLoader:
    # def __init__(self, tknizer, batch_size, max_len, min_len, filename):
    #     self.tknizer = tknizer
    #     self.batch_size = batch_size
    #     self.max_len = max_len
    #     self.min_len = min_len
    #     self.filename = filename
    #     self.stream = open(self.filename, encoding="utf-8")
    #     self.epoch_id = 0

    # def __iter__(self):
    #     lines = self.stream.readlines(BUFFER_SIZE)
    #     if not lines:
    #         self.stream.close()
    #         self.epoch_id += 1
    #         self.stream = open(self.filename, encoding="utf-8")
    #         lines = self.stream.readlines(BUFFER_SIZE)
    #     data = parse_lines(lines[:-1], self.max_len, self.min_len)
    #     random.shuffle(data)
    #     idx = 0
    #     while idx < len(data):
    #         print(1)
    #         yield batchify(data[idx : idx + self.batch_size], self.tknizer)
    #         idx += self.batch_size
    def __init__(self, tknizer, batch_size, max_len, min_len, filename):
        self.tknizer = tknizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.min_len = min_len
        self.filename = filename
        self.epoch_id = 0

    def __iter__(self):
        while True:  # 持续读取文件内容，形成循环
            with open(self.filename, encoding="utf-8") as stream:
                while True:
                    lines = stream.readlines(BUFFER_SIZE)
                    if not lines:  # 如果读取完毕，结束循环，进入新一轮 epoch
                        self.epoch_id += 1
                        break
                    data = parse_lines(lines[:-1], self.max_len, self.min_len)
                    random.shuffle(data)
                    idx = 0
                    while idx < len(data):
                        yield batchify(data[idx : idx + self.batch_size], self.tknizer)
                        idx += self.batch_size
