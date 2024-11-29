import argparse

import torch

from utils.infer_funs import greedy, init_model, mstime, top_k_inc


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dialogue using GPT model")
    parser.add_argument("--model_path", type=str, default="./epoch1_batch_15000", help="Path to the model checkpoint")
    parser.add_argument("--vocab_path", type=str, default="./model/vocab.txt", help="Path to the vocab file")
    parser.add_argument("--max_len", type=int, default=80, help="Maximum length of generated text")
    return parser.parse_args()


def generate_responses(lm_model, lm_vocab, device, questions, max_len):
    for i, q in enumerate(questions):
        start = mstime()
        s = [[w for w in q]]
        s[0].insert(0, "<bos>")

        print(i + 1)
        print("q: ", q)

        # Generate responses using different methods
        r1 = greedy(lm_model=lm_model, lm_vocab=lm_vocab, device=device, s=s, max_len=max_len)
        print("greedy: ", r1)
        r3 = top_k_inc(lm_model=lm_model, lm_vocab=lm_vocab, device=device, s=s, max_len=max_len, k=5)
        print("tk5: ", r3)

        r6 = top_k_inc(lm_model=lm_model, lm_vocab=lm_vocab, device=device, s=s, max_len=max_len, k=50)
        print("tk50: ", r6)
        r7 = top_k_inc(lm_model=lm_model, lm_vocab=lm_vocab, device=device, s=s, max_len=max_len, k=500)
        print("tk500: ", r7)

        print("Time taken: ", mstime() - start)


if __name__ == "__main__":
    args = parse_args()  # Parse command line arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Predefined questions
    questions = [
        # "给你两个角色信息如下：\nA：王明，一个即将毕业的大学生，拥有广泛的兴趣爱好，包括电影、音乐和文学。\nB：张伟，一个广告公司的创意总监，擅长表达和沟通。\n生成他们之间的一段对话，要求对话内容详细丰富。",
        # "请描述一下自然语言处理的主要任务，并简要介绍每个任务的应用场景。",
        # "如何在机器学习模型中处理缺失值？列举几种常用方法。",
        # "请简单介绍一下 GPT 和 BERT 的区别以及它们的应用。",
        # "给定一段中文文本，如何通过深度学习模型进行情感分析？",
        "小明原先有12支铅笔，他拿出了3支铅笔送给同学，剩余的铅笔数可以用减法来计算。即： \n剩余铅笔数 = 原先铅笔数 - 送出的铅笔数 \n所以，剩余的铅笔数 = ",
    ]

    # Load the model and vocab using the parsed paths
    print("Loading model and vocab...")
    lm_model, lm_vocab, lm_args = init_model(args.model_path, device, args.vocab_path)
    print("Model and vocab loaded.")

    # Generate responses for the predefined questions
    generate_responses(lm_model, lm_vocab, device, questions, args.max_len)
