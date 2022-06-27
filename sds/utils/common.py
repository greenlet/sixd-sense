from typing import Union, Tuple

IntOrTuple = Union[int, Tuple[int, int]]


def int_to_tuple(ti: IntOrTuple) -> Tuple[int, int]:
    return ti if type(ti) == tuple else (ti, ti)


