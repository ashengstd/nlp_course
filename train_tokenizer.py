import json
from collections import Counter
from multiprocessing import Pool

BUFSIZE = 10000
ttype = "bpe"


def process(doc):
    res = [w for w in doc]
    return res


def save(cnt, docs, nprocessors):
    res = pool.map(process, docs, len(docs) / nprocessors)
    all_lines = []
    for xs in res:
        all_lines.extend(xs)
    for x in all_lines:
        cnt.update(x)


if ttype == "char":
    cnt: Counter = Counter()
    nprocessors: int = 20
    pool = Pool(nprocessors)
    docs = []
    with open("./data/train.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)["text"]
            if not line:
                continue
            docs.append(line)

    print("vocab")
    with open("./data/vocab.txt", "w", encoding="UTF-8") as f:
        for x, y in cnt.most_common():
            f.write(x + "\t" + str(y) + "\n")
    print("done")
elif ttype == "bpe":
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input="./data/train.txt",
        model_prefix="m",
        vocab_size=32000,
        character_coverage=1.0,
        model_type="bpe",
        num_threads=20,
        user_defined_symbols=["<pad>", "<bos>", "<eos>", "<mask>", "<INST>", "<\INST>", "<SYS>", "<\SYS>"],
    )
elif ttype == "tokenizers":
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    # 初始化分词器
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    # 训练分词器
    trainer = BpeTrainer(
        vocab_size=32000, special_tokens=["<pad>", "<bos>", "<eos>", "<mask>", "<INST>", "<\\INST>", "<SYS>", "<\\SYS>"]
    )
    tokenizer.train(["./data/train.txt"], trainer)

    # 保存分词器
    tokenizer.save("m.json")
