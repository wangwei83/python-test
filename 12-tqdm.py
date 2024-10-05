from tqdm import tqdm

squares = [x**2 for x in tqdm(range(1000))]


from tqdm import tqdm
import time

def process_data(data):
    for item in tqdm(data):
        time.sleep(0.1)  # 模拟处理时间

data = range(100)
process_data(data)



import pandas as pd
from tqdm import tqdm

tqdm.pandas()

df = pd.DataFrame({'a': range(1000)})
df['b'] = df['a'].progress_apply(lambda x: x**2)
