import json
import pathlib


def convert_data(input_file, output_file):
    with open(input_file, encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
        for line in f:
            data = json.loads(line.strip())
            # 将 instruction 和 output 合并为 text 字段
            text = f"{data['instruction']}\n{data['output']}".strip()
            # 创建新的字典格式
            converted_entry = {"text": text, "meta": data.get("meta", "")}
            # 将每个字典转换为单行 JSON 格式
            out_f.write(json.dumps(converted_entry, ensure_ascii=False) + "\n")


path = pathlib.Path("./sft")
for file in path.iterdir():
    if file.suffix == ".txt":
        output_file = file.with_name(file.stem + "_converted.txt")
        convert_data(file, output_file)
