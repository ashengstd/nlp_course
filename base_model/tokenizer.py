import numpy as np
import sentencepiece as spm

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
LS, RS, SP = "<s>", "</s>", ""
BINST, EINST = "<INST>", "</INST>"
BSY, ESY = "<SYS>", "</SYS>"


class Tokenizer:
    def __init__(self, filename, min_occur_cnt, specials=None):
        idx2token = (
            [PAD, UNK, BOS, EOS] + [LS, RS, SP, BINST, EINST, BSY, ESY] + (specials if specials is not None else [])
        )
        for line in open(filename, encoding="utf-8").readlines():
            try:
                token, cnt = line.strip().split()
            except Exception:
                continue
            if int(cnt) >= min_occur_cnt:
                idx2token.append(token)
        self._token2idx = dict(zip(idx2token, range(len(idx2token)), strict=False))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def random_token(self):
        return self.idx2token(1 + np.random.randint(self.size - 1))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)

    def encode(self, x):
        if isinstance(x, list):
            # 如果是列表，逐个将字符串转换为 token ID
            return [self.token2idx(i) for i in x]
        elif isinstance(x, str):
            # 如果是单个字符串，首先分词（如果使用简单的空格分词方式）
            tokens = x.split()  # 这里可以根据需求调整为更复杂的分词方式
            return [self.token2idx(token) for token in tokens]

    def decode(self, x):
        return "".join(self.idx2token(x))


class BpeTokenizer:
    def __init__(self, model_path, specials=None):
        # 初始化特殊符号和BPE模型
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.specials = [PAD, UNK, BOS, EOS, LS, RS, SP, BINST, EINST, BSY, ESY] + (specials if specials else [])

        # 构建特殊符号的索引映射
        self._token2idx = {tok: idx for idx, tok in enumerate(self.specials)}
        self._idx2token = self.specials[:]
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

        # 把BPE模型词汇添加到索引映射中
        for idx in range(self.sp.get_piece_size()):
            token = self.sp.id_to_piece(idx)
            if token not in self._token2idx:
                self._token2idx[token] = len(self._idx2token)
                self._idx2token.append(token)

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def encode(self, text):
        # BPE编码文本
        pieces = self.sp.encode_as_pieces(text)
        return [self._token2idx.get(piece, self._unk_idx) for piece in pieces]

    def decode(self, indices):
        # 将索引解码为文本
        pieces = [self._idx2token[idx] if idx < len(self._idx2token) else UNK for idx in indices]
        return self.sp.decode_pieces(pieces)

    def random_token(self):
        return self._idx2token[1 + np.random.randint(self.size() - 1)]

    def idx2token(self, indices):
        # 将索引转换为token
        if isinstance(indices, list):
            return [self._idx2token[i] if i < len(self._idx2token) else UNK for i in indices]
        return self._idx2token[indices] if indices < len(self._idx2token) else UNK

    def token2idx(self, tokens):
        # 将token转换为索引
        if isinstance(tokens, list):
            return [self._token2idx.get(tok, self._unk_idx) for tok in tokens]
        return self._token2idx.get(tokens, self._unk_idx)


if __name__ == "__main__":
    text = (
        "南京航空航天大学是一所坐落在南京的双一流大学, "
        "Nanjing University of Aeronautics and Astronautics is a double first-class university located in Nanjing."
    )
    # tokenizer = Tokenizer("./model/vocab.txt", min_occur_cnt=50)
    tokenizer = BpeTokenizer("./model/m.model")

    tks = tokenizer.encode(text)
    print(tks)

    dtext = tokenizer.decode(tks)
    print(dtext)
    print(text == dtext)

    sp = spm.SentencePieceProcessor(model_file="./model/m.model")
    tks = sp.encode(text, out_type=int)
    print(tks)

    dtext = sp.decode_pieces(tks)
    print(dtext)
    print(text == dtext)

    print(sp.encode(text, out_type=str))
