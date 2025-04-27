"""
performance evaluation using expression, prefix and numerical value
"""


def convert_expression(test, nums):
    """
    Converts symbolic math expressions into numeric values where applicable.

    Args:
        test (list): Expression list containing symbolic or numeric elements.
        nums (list): Number values corresponding to symbols.

    Returns:
        list or None: Converted expression list or None if conversion fails.
    """
    try:
        return [
            nums[int(item[2:]) % len(nums)] if len(item) > 1 and item[0].lower() == 'n' else
            item[2:].replace('_', '.') if len(item) > 1 and item[0].lower() == 'c' else
            item
            for item in test
        ]
    except (ValueError, IndexError):
        return None


def compute_prefix_expression(pre_fix):
    """
    Compute prefix results.
    """
    st = []
    operators = ["+", "-", "^", "*", "/"]

    for p in pre_fix.reverse():
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


def compute_tree_result(test, target, answer, nums, notation="prefix"):
    """
    Validates a computed mathematical expression against a target.

    Args:
        test (list): Generated expression.
        target (list): Correct expression.
        answer (float): Expected numerical result.
        nums (list): List of available numbers.
        notation (str): Notation type ('prefix', 'infix', 'postfix').

    Returns:
        tuple (bool, bool, list, list): (value_correct, expression_correct, test, target)
    """
    test_expr = convert_expression(test, nums)
    target_expr = convert_expression(target, nums)

    if test_expr is None:
        return False, False

    if test_expr == target_expr:
        return True, True

    try:
        computed_result = compute_prefix_expression(test_expr)
        return (abs(computed_result - answer) < 1e-3, False) if computed_result is not None else (False, False)
    except TypeError:
        return False, False


# Wrappers for specific notation types
def compute_prefix_tree_result(test, target, answer, nums):
    return compute_tree_result(test, target, answer, nums, notation="prefix")
