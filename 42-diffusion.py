# from transformers import AutoImageProcessor, ResNetForImageClassification
# import torch
# from datasets import load_dataset

# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]

# processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# inputs = processor(image, return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# # model predicts one of the 1000 ImageNet classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])

from transformers import BertTokenizer, BertModel

# 下载并加载预训练的 BERT 模型和词汇表
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 输出模型结构
print(model)
