import secrets
import string


def generate_password(length: int = 12) -> str:
    """
    Generate a strong random password.

    Rules:
    - Only uses numbers, lower and upper case ASCII characters, plus a trailing symbol
    - Starts with a letter
    - Avoids ambiguous characters: l, I, 1, 0, O, B, 8
    - Ends with a symbol from: ! # $ & % * = + -

    :param length: Total length of the password (minimum 3).
    :return: A randomly generated password string.
    """
    if length < 8:
        raise ValueError("Password length must be at least 8.")

    AMBIGUOUS = set("lI10OB8")
    SYMBOLS = "!#$&%*=+"

    upper = [c for c in string.ascii_uppercase if c not in AMBIGUOUS]
    letters = [c for c in string.ascii_letters if c not in AMBIGUOUS]
    digits = [c for c in string.digits if c not in AMBIGUOUS]
    middle_chars = letters + digits

    # First character: a letter (not ambiguous)
    first = secrets.choice(upper)

    # Last character: a symbol
    last = secrets.choice(SYMBOLS)

    # Middle characters: letters + digits (not ambiguous), length - 2 chars
    middle = [secrets.choice(middle_chars) for _ in range(length - 2)]

    password = first + "".join(middle) + last
    return password


if __name__ == "__main__":
    for length in (8, 12, 16, 24):
        pwd = generate_password(length)
        print(f"Length {length:2d}: {pwd}")

