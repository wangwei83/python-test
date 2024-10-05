# import tensorflow as tf
# import tensorflow_datasets as tfds
# import matplotlib.pyplot as plt

# # 加载 MNIST 数据集
# (train_data, test_data), info = tfds.load('mnist', split=['train', 'test'], with_info=True, as_supervised=True)

# # 预处理数据
# def preprocess(image, label):
#     image = tf.cast(image, tf.float32) / 255.0
#     label = tf.one_hot(label, depth=10)
#     return image, label

# train_data = train_data.map(preprocess).batch(32)
# test_data = test_data.map(preprocess).batch(32)

# # 构建一个简单的模型
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# # 编译模型
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # 训练模型
# model.fit(train_data, epochs=5)

# # 评估模型
# model.evaluate(test_data)

# # 可视化样本
# for image, label in train_data.take(1):
#     plt.imshow(image[0].numpy().reshape(28, 28), cmap='gray')
#     plt.title(f'Label: {tf.argmax(label[0]).numpy()}')
#     plt.show()


import tensorflow as tf
import numpy as np
import gzip

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # Skip the magic number and dimensions
        f.read(16)
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        data = data.reshape(-1, 28, 28, 1)
        data = data / 255.0  # Normalize to [0, 1]
        return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # Skip the magic number and dimensions
        f.read(8)
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

# 加载训练和测试数据  /home/wangwei83/Desktop/python-test/dataset/mnist/t10k-labels-idx1-ubyte.gz
train_images = load_mnist_images('/home/wangwei83/Desktop/python-test/dataset/mnist/train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('/home/wangwei83/Desktop/python-test/dataset/mnist/train-labels-idx1-ubyte.gz')
test_images = load_mnist_images('/home/wangwei83/Desktop/python-test/dataset/mnist/t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('/home/wangwei83/Desktop/python-test/dataset/mnist/t10k-labels-idx1-ubyte.gz')

print(train_images.shape)
# 将标签转换为 one-hot 编码
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# 构建一个简单的模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=32)

# 评估模型
model.evaluate(test_images, test_labels)

print("Done!")