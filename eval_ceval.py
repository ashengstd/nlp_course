import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from ceval.myEvaluator import MyEvaluator

choices = ["A", "B", "C", "D"]


def main(args):
    # evaluator = myGPT_Evaluator(
    #     choices=choices, model_path=args.model_path, vocab_path=args.vocab_path, device=args.device, k=args.ntrain
    # )
    evaluator = MyEvaluator(choices=choices, model_path=args.model_path, vocab_path=args.vocab_path, device=args.device, k=args.ntrain)
    if args.subject == "all":
        with open("./data/ceval/subjects.json", encoding="utf-8") as file:
            data = json.load(file)  # 解析 JSON 数据
            keys = list(data.keys())
    else:
        keys = [args.subject]
    average_correct_ratio = 0
    keys = [
        "middle_school_geography",
        "high_school_geography",
        "computer_architecture",
        "discrete_mathematics",
        "college_economics",
        "physician",
    ]
    for subject_name in keys:
        run_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        save_result_dir = os.path.join(r"logs", f"MyGPT_{run_date}/{subject_name}")
        Path(save_result_dir).mkdir(parents=True)
        val_file_path = os.path.join("./data/ceval/val", f"{subject_name}_val.csv")
        val_df = pd.read_csv(val_file_path)
        if args.few_shot:
            dev_file_path = os.path.join("./data/ceval/dev", f"{subject_name}_dev.csv")
            dev_df = pd.read_csv(dev_file_path)
            correct_ratio = evaluator.eval_subject(
                subject_name,
                val_df,
                dev_df,
                few_shot=args.few_shot,
                save_result_dir=save_result_dir,
                constrained_decoding=True,
                cot=False,
            )
        else:
            correct_ratio = evaluator.eval_subject(subject_name, val_df, few_shot=args.few_shot, save_result_dir=save_result_dir, constrained_decoding=True)
            average_correct_ratio += correct_ratio
        print("Acc:", correct_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--few_shot", action="store_true", default=False)
    # parser.add_argument("--cot", action="store_true", default=True)
    parser.add_argument("--model_path", type=str, default="./ckpt/epoch1_batch_15000")
    parser.add_argument("--vocab_path", type=str, default="./model/vocab.txt")
    parser.add_argument("--subject", "-s", type=str, default="computer_network")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
