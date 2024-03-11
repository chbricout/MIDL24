from typing import List


def get_parameters(id: int, *params: List[List[any]]):
    div = 1
    correct_id = id - 1
    res = []
    for par in params:
        res.append(par[(correct_id // div) % len(par)])
        div *= len(par)
    return res
