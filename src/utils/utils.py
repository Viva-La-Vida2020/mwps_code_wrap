"""
common Json file loading and saving func.
"""

import json


def load_json(filename):
    """
    load  Json data from filename folder
    """
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_json(data, filename):
    """
    save data to Json file under filename folder
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for d in data:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")
        f.close()
