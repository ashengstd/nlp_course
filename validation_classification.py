import random

from utils.infer_funs import greedy, init_model


# 查找最先出现的目标子字符串
def find_first_occurrence(model_output, target_sequences):
    first_occurrence = None
    first_index = len(model_output)  # 初始化为一个比所有目标序列的索引都大的值

    for target in target_sequences:
        index = model_output.find(target)  # 使用 find 方法查找目标子字符串的首次出现
        if index != -1 and index < first_index:
            first_index = index
            first_occurrence = target

    return first_occurrence, first_index


def read_and_classify(file_path):
    # 定义分类code与名称的映射
    categories = {
        "100": "news_story",
        "101": "news_culture",
        "102": "news_entertainment",
        "103": "news_sports",
        "104": "news_finance",
        "106": "news_house",
        "107": "news_car",
        "108": "news_edu",
        "109": "news_tech",
        "110": "news_military",
        "112": "news_travel",
        "113": "news_world",
        "114": "stock",
        "115": "news_agriculture",
        "116": "news_game",
    }
    name_list = list(categories.values())

    # 初始化统计数据
    stats = {name: {"TP": 0, "FP": 0, "FN": 0} for name in name_list}
    right_num = 0
    total_num = 0

    with open(file_path, encoding="utf-8") as file:
        for line in file:
            fields = line.strip().split("_!_")
            if len(fields) < 5:
                continue

            news_id, category_code, category_name, news_title, news_keywords = fields[:5]

            # 构建prompt
            prompt = (
                f"新闻标题：{news_title}\n新闻关键词：{news_keywords}\n, "
                "它的类别应该是news_story、news_culture、news_entertainment、news_sports、news_finance、news_house、"
                "news_car、news_edu、news_tech、news_military、news_travel、news_world、stock、news_agriculture、news_game之一。"
                "你觉得它的类别是"
            )

            prompt_len = len(prompt)
            inp = [[w for w in prompt]]
            ret = greedy(lm_model, lm_vocab, device, inp, max_len)[prompt_len:]
            print("大预言模型预测结果：", ret)

            # 从预测结果中找到最先出现的目标子字符串
            category_result, _ = find_first_occurrence(ret, name_list)

            if category_result is None:
                category_result = random.choice(name_list)

            print(f"新闻ID：{news_id}\n分类结果：{category_result}\n")

            if category_name == category_result:
                right_num += 1
                stats[category_result]["TP"] += 1
                print("right!")
            else:
                if category_name:
                    stats[category_name]["FP"] += 1
                stats[category_result]["FN"] += 1
            total_num += 1

    # 计算并输出每个类别的准确率、精确率和召回率
    for category, stat in stats.items():
        TP = stat["TP"]
        FP = stat["FP"]
        FN = stat["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = TP / total_num if total_num > 0 else 0
        print(f"类别：{category} - 准确率：{accuracy:.2f}, 精确率：{precision:.2f}, 召回率：{recall:.2f}")

    print("总准确率：", right_num / total_num if total_num > 0 else 0)


if __name__ == "__main__":
    device = 0
    print("loading...")
    m_path = "./epoch1_batch_15000"
    v_path = "./model/vocab.txt"
    lm_model, lm_vocab, lm_args = init_model(m_path, device, v_path)
    print("done.")

    max_len = 100
    prompt: list[str] = []

    val_set_path = "./data/toutiao/val.txt"

    read_and_classify(val_set_path)
