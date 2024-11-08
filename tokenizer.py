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
            except:
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
    if isinstance(self, x):
        return [self.token2idx(i) for i in x]
    return self._token2idx.get(x, self._unk_idx)


def encode(self, x):
    return self.token2idx(x)


def decode(self, x):
    return "".join(self.idx2token(x))


if __name__ == "__main__":
    text = "南京航空航天大学是一所坐落在南京的双一流大学，Nanjing University of Aeronautics and Astronautics is a double first-class university located in Nanjing."
    tokenizer = Tokenizer("./model/vocab.txt", min_occur_cnt=50)

    tks = tokenizer.encode(text)
    print(tks)

    dtext = tokenizer.decode(tks)
    print(dtext)
    print(text == dtext)

    sp = spm.SentencePieceProcessor(model_file="./model/m.model")
    tks = sp.encode_as_pieces(text, out_type=int)
    print(tks)

    dtext = sp.decode_pieces(tks)
    print(dtext)
    print(text == dtext)

    print(sp.encode(text, out_type=str))
