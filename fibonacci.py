def handler(params, context):
    n = params["n"]
    return ''.join(fibonacci_nums(int(n)))

def fibonacci_nums(n):
    if n <= 0:
        sequence = "0"
        return sequence
    sequence = "0, 1"
    count = 2
    n1 = 0
    n2 = 1
    while count <= n:
        next_value = n2 + n1
        sequence += "," + str(next_value)
        n1 = n2
        n2 = next_value
        count += 1
    return sequence

# For command-line testing
if __name__ == "__main__":
    import sys
    sys.set_int_max_str_digits(10000000)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    result = handler({"n": n}, {})
    print("Fibonacci sequence:", result)