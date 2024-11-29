import random


def shuffle_and_split_data(input_file_path, train_file_path, validation_file_path, train_ratio=0.7):
    """
    打乱并划分数据为训练集和验证集，并保存到指定路径
    :param input_file_path: 输入文件路径
    :param train_file_path: 训练集输出文件路径
    :param validation_file_path: 验证集输出文件路径
    :param train_ratio: 训练集占比，默认 70%
    """
    # 读取文件内容
    with open(input_file_path, encoding="utf-8") as file:
        lines = file.readlines()

    # 打乱数据顺序
    random.shuffle(lines)

    # 计算训练集大小
    train_size = int(len(lines) * train_ratio)

    # 将训练集和验证集写入对应文件
    with open(train_file_path, "w", encoding="utf-8") as train_file, open(validation_file_path, "w", encoding="utf-8") as validation_file:
        train_file.writelines(lines[:train_size])
        validation_file.writelines(lines[train_size:])


shuffle_and_split_data("./data/toutiao/toutiao_cat_data.txt", "./data/toutiao/train.txt", "./data/toutiao/validation.txt")


def extract_small_test_set(input_file_path, test_file_path, test_size=300):
    """
    从原始数据集中随机提取指定数量的样本作为测试集
    :param input_file_path: 输入文件路径
    :param test_file_path: 测试集输出文件路径
    :param test_size: 测试集大小，默认为300条
    """
    # 读取文件内容
    with open(input_file_path, encoding="utf-8") as file:
        lines = file.readlines()

    # 打乱数据顺序
    random.shuffle(lines)

    # 提取前test_size条数据作为测试集
    test_set = lines[:test_size]

    # 将测试集写入文件
    with open(test_file_path, "w", encoding="utf-8") as test_file:
        test_file.writelines(test_set)


extract_small_test_set("./data/toutiao/validation.txt", "./data/toutiao/test_set.txt")
