import torch
import torch.nn.functional as F
from torch import nn

from base_model.transformer import Embedding, LearnedPositionalEmbedding, SelfAttentionMask, TransformerLayer
from base_model.utils import LayerNorm, gelu


class MyGPT(nn.Module):
    def __init__(self, local_rank, vocab, embed_dim, ff_embed_dim, num_heads, dropout, layers):
        super().__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim

        self.tok_embed = Embedding(self.vocab.size, embed_dim, self.vocab.padding_idx)
        self.pos_embed = LearnedPositionalEmbedding(embed_dim, device=local_rank)

        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout))
        self.emb_layer_norm = LayerNorm(embed_dim)
        self.one_more = nn.Linear(embed_dim, embed_dim)
        self.one_more_layer_norm = LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, self.vocab.size)

        self.attn_mask = SelfAttentionMask(device=local_rank)

        self.dropout = dropout
        self.device = local_rank

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.one_more.bias, 0.0)
        nn.init.normal_(self.one_more.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -torch.log(torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0) if avg else torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        ppl = 2**cost
        return cost.mean(), cost.sum().item(), ppl.sum().item()

    def ppl(self, truth, inp, msk):
        seq_len, bsz = inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        padding_mask = torch.eq(truth, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers:
            x, _, _ = layer(x, self_padding_mask=padding_mask, self_attn_mask=self_attn_mask)

        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        pred = torch.softmax(self.out_proj(x), -1)
        _, pred_y = pred.max(-1)
        tot_tokens = msk.float().sum().item()
        acc = (torch.eq(pred_y, truth).float() * msk).sum().item()
        loss, nll, ppl = self.nll_loss(pred, truth, msk)
        return acc, nll, ppl, tot_tokens, bsz

    def work(self, inp):
        seq_len, bsz = inp.size()

        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(inp, self.vocab._padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers:
            x, _, _ = layer(x, self_padding_mask=padding_mask, self_attn_mask=self_attn_mask)

        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        probs = torch.softmax(self.out_proj(x), -1)
        _, pred_y = probs.max(-1)

        return probs, pred_y

    def work_incremental(self, inp, incremental_state=None):
        seq_len, bsz = inp.size()
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        padding_mask = torch.eq(inp, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        if incremental_state is None:
            self_attn_mask = self.attn_mask(seq_len)
            incremental_state = {}
        else:
            x = x[-1, :, :].unsqueeze(0)
            self_attn_mask = None

        for layer in self.layers:
            x, _, _ = layer.work_incremental(
                x, self_padding_mask=padding_mask, self_attn_mask=self_attn_mask, incremental_state=incremental_state
            )

        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        probs = torch.softmax(self.out_proj(x), -1)

        _, pred_y = probs.max(-1)
        return probs, pred_y, incremental_state

    def forward(self, truth, inp, msk):
        seq_len, bsz = inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(truth, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers:
            x, _, _ = layer(x, self_padding_mask=padding_mask, self_attn_mask=self_attn_mask)

        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        pred = torch.softmax(self.out_proj(x), -1)

        loss, nll, ppl = self.nll_loss(pred, truth, msk)
        _, pred_y = pred.max(-1)
        tot_tokens = msk.float().sum().item()
        acc = (torch.eq(pred_y, truth).float() * msk).sum().item()
        return (pred_y, truth), loss, acc, nll, ppl, tot_tokens, bsz

    def generate_logits(self, input_ids, attention_mask=None):
        """
        生成给定输入的 logits（未归一化的概率）。

        Args:
            input_ids (torch.Tensor): 输入的 token IDs，形状为 (seq_len, batch_size)。
            attention_mask (torch.Tensor, optional): 可选的注意力掩码，形状为 (seq_len, batch_size)。

        Returns:
            torch.Tensor: 输出的 logits，形状为 (seq_len, batch_size, vocab_size)。
        """
        seq_len, bsz = input_ids.size()

        # 获取自注意力掩码
        self_attn_mask = self.attn_mask(seq_len)

        # 嵌入 + 位置编码 + 层归一化
        x = self.tok_embed(input_ids) + self.pos_embed(input_ids)
        x = self.emb_layer_norm(x)

        # 如果 attention_mask 提供，将其转为 padding_mask
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()  # 将 0/1 掩码转换为布尔形式
        else:
            padding_mask = torch.eq(input_ids, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        # 通过 Transformer 层
        for layer in self.layers:
            x, _, _ = layer(x, self_padding_mask=padding_mask, self_attn_mask=self_attn_mask)

        # 层归一化 + 非线性激活
        x = self.one_more_layer_norm(gelu(self.one_more(x)))

        # 输出 logits
        logits = self.out_proj(x)
        return logits
