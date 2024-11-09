import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Embedding, LearnedPositionalEmbedding, SelfAttentionMask, TransformerLayer
from utils import LayerNorm, gelu


class MyGPT(nn.Module):
    def __init__(self, device, vocab, embed_dim, ff_embed_dim, num_heads, dropout, num_layers):
        super().__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim

        self.token_embedding = Embedding(self.vocab.size, embed_dim, self.vocab.padding_idx)
        self.position_embedding = LearnedPositionalEmbedding(embed_dim, device=device)

        self.layers = nn.ModuleList(
            [TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.embedding_layer_norm = LayerNorm(embed_dim)
        self.intermediate_linear = nn.Linear(embed_dim, embed_dim)
        self.intermediate_layer_norm = LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, self.vocab.size)

        self.attention_mask = SelfAttentionMask(device=device)

        self.dropout = dropout
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.intermediate_linear.bias, 0.0)
        nn.init.normal_(self.intermediate_linear.weight, std=0.02)
        nn.init.constant_(self.output_projection.bias, 0.0)
        nn.init.normal_(self.output_projection.weight, std=0.02)

    def nll_loss(self, predictions, targets, mask, average=True):
        log_probs = -torch.log(torch.gather(predictions, 2, targets.unsqueeze(2)))
        log_probs = log_probs.view(targets.shape)
        mask = mask.view(targets.shape)
        log_probs = torch.sum(log_probs * mask, 0) / torch.sum(mask, 0) if average else torch.sum(log_probs * mask, 0)
        log_probs = log_probs.view((targets.size(1), -1))
        perplexity = 2**log_probs
        return log_probs.mean(), log_probs.sum().item(), perplexity.sum().item()

    def forward_pass(self, inputs, targets=None, mask=None, incremental_state=None):
        seq_len, batch_size = inputs.size()
        attention_mask = self.attention_mask(seq_len) if incremental_state is None else None
        x = self.token_embedding(inputs) + self.position_embedding(inputs)
        x = self.embedding_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(targets if targets is not None else inputs, self.vocab.padding_idx)
        padding_mask = None if not padding_mask.any() else padding_mask

        for layer in self.layers:
            if incremental_state is None:
                x, _, _ = layer(x, self_padding_mask=padding_mask, self_attn_mask=attention_mask)
            else:
                x, _, _ = layer.work_incremental(
                    x[-1, :, :].unsqueeze(0),
                    self_padding_mask=padding_mask,
                    self_attn_mask=attention_mask,
                    incremental_state=incremental_state,
                )

        x = self.intermediate_layer_norm(gelu(self.intermediate_linear(x)))
        probabilities = torch.softmax(self.output_projection(x), -1)
        return probabilities

    def calculate_ppl(self, targets, inputs, mask):
        predictions = self.forward_pass(inputs, targets)
        _, predicted_tokens = predictions.max(-1)
        total_tokens = mask.float().sum().item()
        accuracy = (torch.eq(predicted_tokens, targets).float() * mask).sum().item()
        loss, nll, perplexity = self.nll_loss(predictions, targets, mask)
        return accuracy, nll, perplexity, total_tokens, inputs.size(1)

    def generate(self, inputs):
        probabilities = self.forward_pass(inputs)
        _, predicted_tokens = probabilities.max(-1)
        return probabilities, predicted_tokens

    def generate_incremental(self, inputs, incremental_state=None):
        probabilities = self.forward_pass(inputs, incremental_state=incremental_state)
        _, predicted_tokens = probabilities.max(-1)
        return probabilities, predicted_tokens, incremental_state

    def forward(self, targets, inputs, mask):
        predictions = self.forward_pass(inputs, targets)
        loss, nll, perplexity = self.nll_loss(predictions, targets, mask)
        _, predicted_tokens = predictions.max(-1)
        total_tokens = mask.float().sum().item()
        accuracy = (torch.eq(predicted_tokens, targets).float() * mask).sum().item()
        return (predicted_tokens, targets), loss, accuracy, nll, perplexity, total_tokens, inputs.size(1)
