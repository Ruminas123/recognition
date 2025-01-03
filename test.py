import random

def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


for i in range(random.randint(1, 10)):
    print(i)