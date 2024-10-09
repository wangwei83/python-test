
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 生成示例数据
# data = np.random.normal(loc=0, scale=1, size=1000)  # 正态分布数据
data = np.random.exponential(scale=1, size=1000)  # 非正态分布数据

# 将数据转换为 DataFrame
df = pd.DataFrame(data, columns=['Column1'])

# 绘制直方图和QQ图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['Column1'], kde=True)
plt.title('Histogram')

plt.subplot(1, 2, 2)
stats.probplot(df['Column1'], dist="norm", plot=plt)
plt.title('QQ Plot')

plt.show()

# 进行Shapiro-Wilk正态性检验
stat, p = stats.shapiro(df['Column1'])
print('Statistics=%.3f, p=%.3f' % (stat, p))

# 解释结果
alpha = 0.05
if p > alpha:
    print('样本看起来符合正态分布 (接受H0)')
else:
    print('样本看起来不符合正态分布 (拒绝H0)')
