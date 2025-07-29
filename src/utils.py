import math

# math module has not yet implemented a sign function
# see https://bugs.python.org/msg59154
sign = lambda x: int(math.copysign(1.0, x))  # noqa: E731


def bezout_2d(a: int, b: int) -> tuple[int, int]:
    sgn_a, sgn_b = sign(a), sign(b)
    a, b = abs(a), abs(b)

    prev_r, r = a, b
    prev_x, x = 1, 0
    prev_y, y = 0, 1

    while r != 0:
        q = prev_r // r

        prev_r, r = r, prev_r - q * r
        prev_x, x = x, prev_x - q * x
        prev_y, y = y, prev_y - q * y

    return sgn_a * prev_x, sgn_b * prev_y


def bezout(integers: list[int]) -> list[int]:
    a, b = integers[0], integers[1]
    x, y = bezout_2d(a, b)

    coeffs: list[int] = [x, y]
    for i in range(2, len(integers)):
        a = a * x + b * y
        b = integers[i]
        x, y = bezout_2d(a, b)

        for j in range(i):
            coeffs[j] *= x

        coeffs.append(y)

    return coeffs
