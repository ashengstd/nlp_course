import os
import re
import string

import torch
from tqdm import tqdm

from data import s2t
from mygpt import MyGPT
from tokenizer import Tokenizer


class Evaluator:
    def __init__(self, choices, model_path, vocab_path, device, k=-1):
        self.choices = choices
        self.model_path = model_path
        self.k = k
        self.puncs = list(string.punctuation)
        self.init_model(model_path, device, vocab_path)
        self.device = device

    def init_model(self, m_path, device, vocab):
        ckpt = torch.load(m_path, map_location="cpu")
        self.args = ckpt["args"]
        self.tokenizer = Tokenizer(vocab, min_occur_cnt=self.args.min_occur_cnt, specials=[])
        self.model = MyGPT(
            device,
            self.tokenizer,
            self.args.embed_dim,
            self.args.ff_embed_dim,
            self.args.num_heads,
            self.args.dropout,
            self.args.layers,
        )
        self.model.load_state_dict(ckpt["model"])
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def top_k_inc(self, lm_model, lm_vocab, device, s, k, max_len, history=None):
        incremental_state = None
        # if history is not None:
        # s = [history+sen for sen in s]
        x, m = s2t(s, lm_vocab)
        x = x.to(device)
        res = []
        for l in range(max_len):
            probs, pred, incremental_state = lm_model.work_incremental(x, incremental_state)
            next_tk = []
            for i in range(len(s)):
                logits = probs[len(s[i]) - 1, i] if l == 0 else probs[0, i]
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

    def format_example(self, line, include_answer=True):
        example = line["question"]
        # print(example)
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        example += "\n答案："
        if include_answer:
            example += f'{line["answer"]}\n\n'
        return example

    def generate_few_shot_prompt(self, subject, dev_df):
        prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(dev_df.iloc[i, :])
        return prompt

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None):
        correct_num = 0
        if save_result_dir:
            if few_shot:
                result = []
            score = []
        history = self.generate_few_shot_prompt(subject_name, dev_df) if few_shot else []
        answers = list(test_df["answer"])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False)
            question = [[q for q in question]]
            response = self.top_k_inc(
                lm_model=self.model,
                lm_vocab=self.tokenizer,
                s=question,
                k=5,
                max_len=256,
                history=history,
                device=self.device,
            )
            response = response.strip()
            print(response)
            # For ChatGLM, we use answer extraction in answer-only mode too.
            ans = self.extract_first_choice(response)
            if ans == answers[row_index]:
                correct_num += 1
                correct = 1
            else:
                correct = 0
            if save_result_dir:
                if few_shot:
                    result.append(response)
                score.append(correct)
        correct_ratio = 100 * correct_num / len(answers)

        if save_result_dir:
            if few_shot:
                test_df["model_output"] = result
            test_df["correctness"] = score
            test_df.to_csv(os.path.join(save_result_dir, f"{subject_name}_test.csv"))

        return correct_ratio

    def extract_first_choice(self, text):
        # 使用正则表达式寻找第一个“A”、“B”、“C”或“D”
        match = re.search(r"[ABCD]", text)
        return match.group(0) if match else None

    def normalize_answer(self, s):
        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(self.puncs)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))

    def exact_match(self, pred, target):
        return self.normalize_answer(pred) == self.normalize_answer(target)
