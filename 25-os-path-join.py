import os

# 假设 config 是一个包含输出目录、数据集和子数据集信息的对象
# 也可以来自get_argparse()函数或其他方式，get_argparse()由argparse库提供并衍生出来
class Config:
    output_dir = '/path/to/output'
    dataset = 'my_dataset'
    subdataset = 'sub_dataset'

config = Config()

# 构建训练输出目录的路径
train_output_dir = os.path.join(config.output_dir, 'trainings', 
                                config.dataset, config.subdataset)

print(train_output_dir)
