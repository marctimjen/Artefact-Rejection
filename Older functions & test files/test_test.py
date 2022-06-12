

def tester(a, b):
    """
    Find the sum(a, b).

    Examples:
    >>> tester(1, 1)
    2
    """

    return a + b


class Test:
    """
    >>> Test.multiply_by_3(Test,2)
    7
    """
    def __init__(self, number):
        self._number=number

    _THREE = 3
    def multiply_by_3(self, x):
        return x*self._THREE

if __name__ == "__main__":
    import doctest
    doctest.testmod()
