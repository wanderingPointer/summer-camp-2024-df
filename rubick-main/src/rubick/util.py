def upDiv(x: int, y: int) -> int:
    return x // y + (0 if x % y == 0 else 1)
