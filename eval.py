import argparse
import json
import os
import time

import pandas as pd
from evaluator import Evaluator

choices = ["A", "B", "C", "D"]


def main(args):
    evaluator = Evaluator(
        choices=choices, model_path=args.model_path, vocab_path=args.vocab_path, device=args.device, k=args.ntrain
    )
    with open("subjects.json", encoding="utf-8") as file:
        data = json.load(file)  # 解析 JSON 数据
    keys = list(data.keys()) if args.subject == "all" else [args.subject]
    for subject_name in keys:
        if not os.path.exists(r"logs"):
            os.mkdir(r"logs")
        run_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        save_result_dir = os.path.join(r"logs", f"MyGPT_{run_date}")
        os.mkdir(save_result_dir)
        print(subject_name)
        val_file_path = os.path.join("./val", f"{subject_name}_val.csv")
        val_df = pd.read_csv(val_file_path)
        if args.few_shot:
            dev_file_path = os.path.join("./dev", f"{subject_name}_dev.csv")
            dev_df = pd.read_csv(dev_file_path)
            correct_ratio = evaluator.eval_subject(
                subject_name, val_df, dev_df, few_shot=args.few_shot, save_result_dir=save_result_dir
            )
        else:
            correct_ratio = evaluator.eval_subject(
                subject_name, val_df, few_shot=args.few_shot, save_result_dir=save_result_dir
            )
        print("Acc:", correct_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--few_shot", action="store_true", default=True)
    parser.add_argument("--model_path", type=str, default="./ckpt/epoch0_batch_15000")
    parser.add_argument("--vocab_path", type=str, default="./model/vocab.txt")
    parser.add_argument("--subject", "-s", type=str, default="computer_network")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
