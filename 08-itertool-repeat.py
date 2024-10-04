import itertools

# 无限重复生成 None
repeater = itertools.repeat(None)

# 输出前五个值
for _ in range(5):
    print(next(repeater))