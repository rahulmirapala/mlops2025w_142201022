from utils import check_numeric

def add(a, b):
    check_numeric(a)
    check_numeric(b)
    return a + b