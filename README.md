# NLP - MyGPT

本仓库为 NLP 课程的实验代码

# Dataset

先根据课程提供的链接下载数据集，然后解压到 `data` 文件夹下
最后结构为：

```
> data
    > ceval
    > pretrain
    > sft
        > subject.json
        > dev test val dir
```

# 环境配置

## 安装 uv

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 同步环境

```shell
uv sync
```

## 部分注意事项

- vscode 或者 shell 情况下默认没有配置`PYTHONPATH`，需执行

```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

- SFT 数据集格式不同，需要先执行

```shell
python utils/convert_sft.py
```

# Train Tokenizer

```shell
bash step01.sh
```

# Pretrain

```shell
# 可在step02.sh中修改参数
bash step02.sh
```

# Supervised Fine-tuning

```shell
# 可在step03.sh中修改参数，以及pretrain的模型的路径
bash step03.sh
```

# C-Eval

```shell
python eval.py (--model_path {model_path} --vocab_path {vocab_path} --subject {subject} --ntrain {ntrain} --fewshot)
```
