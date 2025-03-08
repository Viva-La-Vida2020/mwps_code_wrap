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