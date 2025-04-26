def bezout(a: int, b: int) -> tuple[int, int]:
    prev_r, r = a, b
    prev_x, x = 1, 0
    prev_y, y = 0, 1

    while r != 0:
        q = prev_r // r

        prev_r, r = r, prev_r - q * r
        prev_x, x = x, prev_x - q * x
        prev_y, y = y, prev_y - q * y

    return prev_x, prev_y


def gcd(a: int, b: int) -> int:
    x, y = bezout(a, b)
    return a * x + b * y
