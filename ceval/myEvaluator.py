import contextlib
import os
import re

import numpy as np
import torch
from tqdm import tqdm

from base_model.mygpt import MyGPT
from base_model.tokenizer import BOS, Tokenizer
from ceval.evaluator import Evaluator


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs[-1, 0, :], dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class MyEvaluator(Evaluator):
    def __init__(self, model_path, vocab_path, choices, device, k=-1) -> None:
        self.device = device
        self.tokenizer = Tokenizer(vocab_path, min_occur_cnt=1, specials=[])
        self.model = MyGPT(
            local_rank=0, vocab=self.tokenizer, embed_dim=768, ff_embed_dim=3072, num_heads=12, dropout=0.2, layers=12
        )
        ckpt = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.model.cuda()
        self.choices = choices
        self.k = k
        self.patterns = [
            "答案是?\s?([ABCD])",
            "答案是?\s?：([ABCD])",
            "答案是?\s?:([ABCD])",
            "答案应该?是\s?([ABCD])",
            "答案应该?选\s?([ABCD])",
            "答案为\s?([ABCD])",
            "选择\s?([ABCD])",
            "只有选?项?\s?([ABCD])\s?是?对",
            "只有选?项?\s?([ABCD])\s?是?错",
            "只有选?项?\s?([ABCD])\s?不?正确",
            "只有选?项?\s?([ABCD])\s?错误",
            "说法不?对选?项?的?是\s?([ABCD])",
            "说法不?正确选?项?的?是\s?([ABCD])",
            "说法错误选?项?的?是\s?([ABCD])",
            "([ABCD])\s?是正确的",
            "([ABCD])\s?是正确答案",
            "选项\s?([ABCD])\s?正确",
            "所以答\s?([ABCD])",
            "1.\s?([ABCD])[.。$]?$",
            "所以\s?([ABCD][.。$]?$)",
            "所有\s?([ABCD][.。$]?$)",
            "[\s，：:,]([ABCD])[。，,\.]?$",
            "[\s，,：:][故即]([ABCD])[。\.]?$",
            "[\s，,：:]因此([ABCD])[。\.]?$",
            "[是为。]\s?([ABCD])[。\.]?$",
            "因此\s?([ABCD])[。\.]?$",
            "显然\s?([ABCD])[。\.]?$",
            "1.\s?(.*?)$",
            "答案是\s?(\S+)(?:。|$)",
            "答案应该是\s?(\S+)(?:。|$)",
            "答案为\s?(\S+)(?:。|$)",
        ]

    def format_example(self, line, include_answer=True, cot=False):
        example = line["question"]
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        if include_answer:
            if cot:
                example += (
                    "\n答案：让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{line['answer']}。\n\n"
                )
            else:
                example += "\n答案：" + line["answer"] + "\n\n"
        else:
            if cot:
                example += "\n答案：让我们一步一步思考，\n1."
            else:
                example += "\n答案："
        return example

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)
        return prompt

    def generate(
        self, prompt: str, max_gen_len: int, temperature: float = 0.8, top_p: float = 0.95, return_logits: bool = False
    ) -> list[str]:
        # params = self.model.params
        prompt = BOS + prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_size = len(prompt_tokens)
        total_len = max_gen_len + prompt_size

        tokens = torch.full((1, total_len), self.tokenizer._padding_idx).cuda().long()
        tokens[0, :prompt_size] = torch.tensor(prompt_tokens).long()
        input_text_mask = tokens != self.tokenizer._padding_idx
        if return_logits:
            return self.model.work(tokens[:, :prompt_size])[0]
        for cur_pos in range(prompt_size, total_len):
            logits = self.model.work(tokens[:, :cur_pos])[0]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

        decoded = []
        for _, t in enumerate(tokens.tolist()):
            t = t[: prompt_size + max_gen_len]
            with contextlib.suppress(ValueError):
                t = t[: t.index(self.tokenizer._eos_idx)]
            decoded.append(self.tokenizer.decode(t))
        return decoded

    def extract_model_answer(self, text, a, b, c, d):
        option_str = re.escape("A. " + a + "\nB. " + b + "\nC. " + c + "\nD. " + d)
        match = re.search(rf"{option_str}([\s\S]*)$", text)
        if match:
            return match.group(1)
        else:
            return None

    def extract_answer_option(self, text):
        match = re.findall(r"(让我们一步一步思考[\s\S]+?)(?:(?=让我们一步一步思考)|$)", text)
        text = match[0]
        regexes = [re.compile(pattern) for pattern in self.patterns]
        for regex in regexes:
            match = regex.search(text)
            if match:
                return match.group(1)
        return None

    def answer_str(self, answer, a, b, c, d):
        ans_dict = {"A": a, "B": b, "C": c, "D": d}
        return ans_dict[answer]

    def extract_answer(self, row, output):
        pred = {"A": 0, "B": 1, "C": 2, "D": 3}
        correct_answer_str = self.answer_str(row["answer"], row["A"], row["B"], row["C"], row["D"])
        generate_answer = self.extract_model_answer(str(output), row["A"], row["B"], row["C"], row["D"])
        if not generate_answer:
            return None, 0
        model_answer = self.extract_answer_option(generate_answer)
        if row["answer"] == model_answer or correct_answer_str == model_answer:
            return pred.get(model_answer, model_answer), 1
        else:
            return pred.get(model_answer, model_answer), 0

    def eval_subject(
        self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None, cot=False, **kwargs
    ):
        result = []
        score = []
        few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot) if few_shot else ""
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot)
            full_prompt = few_shot_prompt + question
            output = self.generate(
                full_prompt,
                max_gen_len=kwargs.get("max_gen_len", 512),
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95),
                return_logits=not cot,
            )
            if cot:
                assert isinstance(output[0], str)
                output = output[0]
                pred, correct = self.extract_answer(row, output)
            else:
                assert output.shape[0] == 1
                logits = output.flatten()
                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(
                            [
                                logits[self.tokenizer.encode("A")],
                                logits[self.tokenizer.encode("B")],
                                logits[self.tokenizer.encode("C")],
                                logits[self.tokenizer.encode("D")],
                            ]
                        ),
                        dim=0,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
                correct = 1 if pred == row["answer"] else 0
            result.append(pred)
            score.append(correct)
        correct_ratio = 100 * sum(score) / len(score)

        if save_result_dir:
            test_df["model_output"] = result
            test_df["correctness"] = score
            test_df.to_csv(os.path.join(save_result_dir, f"{subject_name}_test.csv"), encoding="utf-8", index=False)
        return correct_ratio
