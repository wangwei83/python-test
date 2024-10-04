# permutation and combination 排列和组合 
import itertools

# 定义一个列表
data = ['A', 'B', 'C']

# 生成所有排列  come from itertools
permutations = list(itertools.permutations(data))
print("所有排列:")
for p in permutations:
    print(p)

# 生成所有组合（长度为2）
combinations = list(itertools.combinations(data, 2))
print("\n所有组合（长度为2）:")
for c in combinations:
    print(c)

# 生成所有组合（长度为2，允许重复）
combinations_with_replacement = list(itertools.combinations_with_replacement(data, 2))
print("\n所有组合（长度为2，允许重复）:")
for cr in combinations_with_replacement:
    print(cr)