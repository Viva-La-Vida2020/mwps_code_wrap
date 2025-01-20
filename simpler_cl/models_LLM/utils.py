import copy
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


def out_expression_list(test, nums):
    try:
        res = []
        for i in test:
            if len(i) > 1 and i[0].lower() == 'n':
                res.append(nums[int(i[2:])%len(nums)])
            elif len(i) > 1 and i[0].lower() == 'c':
                res.append(i[2:].replace('_', '.'))
            else:
                res.append(i)
        return res
    except:
        return None


def compute_prefix_expression(pre_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = copy.deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None


def compute_prefix_result(test, tar, ans, nums):
    test = out_expression_list(test, nums)
    tar = out_expression_list(tar, nums)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - ans) < 1e-3:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar