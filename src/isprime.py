def is_prime(n):
    """
    Check if a number is prime
    Args:
        n: integer number to check
    Returns:
        bool: True if number is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python isprime.py <number>")
        sys.exit(1)
        
    try:
        number = int(sys.argv[1])
        result = is_prime(number)
        print(f"{number} {'is' if result else 'is not'} a prime number")
    except ValueError:
        print("Please provide a valid integer number")
        sys.exit(1)
