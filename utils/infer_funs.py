import copy
import time

import torch

from base_model.mygpt import MyGPT
from base_model.tokenizer import Tokenizer
from dpo_model.config import gpt_config
from utils.data import s2t


def mstime():
    return int(round(time.time() * 1000))


def init_model(m_path, device, vocab):
    ckpt = torch.load(m_path, map_location="cpu")
    lm_args = ckpt["args"]
    lm_vocab = Tokenizer(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    lm_model = MyGPT(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers)
    lm_model.load_state_dict(ckpt["model"])
    lm_model = lm_model.to(device)
    lm_model.eval()
    return lm_model, lm_vocab, lm_args


def init_dpo_model(m_path, device, vocab):
    ckpt = torch.load(m_path, map_location="cpu")
    lm_vocab = Tokenizer(vocab, min_occur_cnt=10, specials=[])
    lm_model = MyGPT(
        local_rank=gpt_config["local_rank"],
        vocab=gpt_config["vocab"],
        embed_dim=gpt_config["embed_dim"],
        ff_embed_dim=gpt_config["ff_embed_dim"],
        num_heads=gpt_config["num_heads"],
        dropout=gpt_config["dropout"],
        layers=gpt_config["layers"],
    )
    lm_model.load_state_dict(ckpt)
    lm_model = lm_model.to(device)
    lm_model.eval()
    return lm_model, lm_vocab


@torch.no_grad()
def greedy(lm_model, lm_vocab, device, s, max_len):
    # 将 prompt 转换为模型输入格式
    prompt = copy.deepcopy(s)
    x, m = s2t(prompt, lm_vocab)  # 假设 s2t 是一个将文本转为 token 的函数
    x = x.to(device)

    generated_tokens = []  # 用于存储生成的 tokens
    for _ in range(max_len):
        # 获取模型的预测输出
        logits, __ = lm_model.work(x)  # 假设 work 是模型的前向传播函数

        # 获取最后一个时间步的输出 logits，选择最大概率的 token
        predicted_token = torch.argmax(logits[-1, 0, :], dim=-1)  # 选择 batch_size 为 1 时的最大概率 token
        next_token = lm_vocab.idx2token(predicted_token.item())  # 将 token ID 转换为词汇表中的 token
        # 添加生成的 token 到生成序列中
        generated_tokens.append(next_token)

        # 如果生成了结束符 <eos>，则提前停止
        if next_token == "<eos>":
            break

        # 将生成的 token 添加到输入序列中
        prompt[0].append(next_token)  # 假设 prompt 是一个列表（包含已生成的 tokens）
        # 更新输入序列
        x, m = s2t(prompt, lm_vocab)  # 更新输入格式
        x = x.to(device)

    # 拼接生成的 token 为文本
    generated_text = "".join(prompt[0])

    return generated_text.split("<bos>")[1] if "<bos>" in generated_text else generated_text


@torch.no_grad()
def top_k_inc(lm_model, lm_vocab, device, s, max_len, k=10):
    prompt = copy.deepcopy(s)
    # 将 prompt 转换为模型输入格式
    x, m = s2t(prompt, lm_vocab)  # 假设 s2t 是一个将文本转为 token 的函数
    x = x.to(device)

    # 初始化增量状态
    incremental_state = None

    generated_tokens = []  # 用于存储生成的 tokens
    for _ in range(max_len):
        # 获取模型的预测输出
        logits, _, incremental_state = lm_model.work_incremental(x, incremental_state)  # 使用增量解码

        # 对 logits 进行 softmax，得到每个 token 的概率分布
        probs = logits[-1, 0, :]
        # 获取前 k 个 token 的概率和索引
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
        top_k_probs = top_k_probs / torch.sum(top_k_probs)  # 归一化
        # 从前 k 个 token 中进行采样，采用随机选择
        next_token_idx = torch.multinomial(top_k_probs, 1)  # 从 top-k 中进行采样
        next_token = top_k_indices[next_token_idx.item()]  # 获取采样到的 token ID

        # 将生成的 token 转换为实际的词
        next_token_str = lm_vocab.idx2token(next_token.item())

        # 如果生成了结束符 <eos>，则提前停止
        if next_token_str == "<eos>":
            break

        # 将生成的 token 添加到生成序列中
        generated_tokens.append(next_token_str)
        # 将生成的 token 添加到输入序列中
        prompt[0].append(next_token_str)  # 假设 prompt 是二维 list, prompt[0] 代表句子
        # 更新输入序列
        x, m = s2t(prompt, lm_vocab)  # 更新输入格式
        x = x.to(device)

    # 拼接生成的 token 为文本
    generated_text = "".join(prompt[0])

    return generated_text.split("<bos>")[1] if "<bos>" in generated_text else generated_text


@torch.no_grad()
def top_k_inc1(lm_model, lm_vocab, device, s, k, max_len):
    incremental_state = None
    x, m = s2t(s, lm_vocab)
    x = x.to(device)
    res = []
    for i in range(max_len):
        probs, pred, incremental_state = lm_model.work_incremental(x, incremental_state)
        next_tk = []
        for i in range(len(s)):
            logits = probs[len(s[i]) - 1, i] if i == 0 else probs[0, i]
            ps, idx = torch.topk(logits, k=k)
            ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples=1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))
        s_ = []
        bidx = [1] * len(s)
        for idx, (sent, t) in enumerate(zip(s, next_tk, strict=False)):
            if t == "<eos>":
                res.append(sent)
                bidx[idx] = 0
            else:
                s_.append(sent + [t])
        if not s_:
            break
        s = s_
        x, m = s2t(s, lm_vocab)
        x = x.to(device)
        bidx = torch.BoolTensor(bidx).to(device)
        incremental_state["bidx"] = bidx
    res += s_
    r = "".join(res[0])
    return r.split("<bos>")[1] if "<bos>" in r else r
