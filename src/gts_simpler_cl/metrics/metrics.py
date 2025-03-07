import copy


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


def compute_expression(expression, notation="prefix"):
    """
    Evaluates a mathematical expression given in prefix, infix, or postfix notation.

    Args:
        expression (list): List of string elements representing a math expression.
        notation (str): Type of notation ('prefix', 'infix', 'postfix').

    Returns:
        float or None: Computed result or None if evaluation fails.
    """
    operators = {"+", "-", "^", "*", "/"}
    stack = []

    if notation == "prefix":
        expression = list(reversed(expression))
    elif notation == "infix":
        try:
            expr_str = ''.join(expression).replace('[', '(').replace(']', ')').replace('^', '**')
            return eval(expr_str)
        except (SyntaxError, ZeroDivisionError, NameError):
            return None

    for token in expression:
        if token not in operators:
            try:
                stack.append(eval(str(token)))
            except (SyntaxError, NameError):
                return None
        elif len(stack) > 1:
            a, b = stack.pop(), stack.pop() if notation != "postfix" else (stack.pop(), stack.pop())
            if token == "+":
                stack.append(a + b)
            elif token == "-":
                stack.append(a - b)
            elif token == "*":
                stack.append(a * b)
            elif token == "/":
                if b == 0:
                    return None
                stack.append(a / b)
            elif token == "^":
                stack.append(a ** b)
        else:
            return None

    return stack[0] if len(stack) == 1 else None


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
        computed_result = compute_expression(test_expr, notation)
        return (abs(computed_result - answer) < 1e-3, False) if computed_result is not None else (False, False)
    except TypeError:
        return False, False


# Wrappers for specific notation types
def compute_prefix_tree_result(test, target, answer, nums):
    return compute_tree_result(test, target, answer, nums, notation="prefix")


def compute_infix_tree_result(test, target, answer, nums):
    return compute_tree_result(test, target, answer, nums, notation="infix")


def compute_postfix_tree_result(test, target, answer, nums):
    return compute_tree_result(test, target, answer, nums, notation="postfix")
