import random

random.seed(42)
print(random.random())  # 每次运行输出相同
print(random.randint(1, 10))  # 每次运行输出相同

import random

random.seed(1)
print(random.random())  # 对于种子1，输出相同

random.seed(2)
print(random.random())  # 对于种子2，输出相同
