import csv
import json


def load_json(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_json(data, filename):
    f = open(filename, 'w', encoding='utf-8')
    for d in data:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()


def load_csv(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        # 创建一个CSV读取器对象
        csv_reader = csv.reader(file)
        # 遍历每一行数据
        for row in csv_reader:
            data.append(row)
    return data