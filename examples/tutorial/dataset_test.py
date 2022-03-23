import numpy as np
import luojianet_ms.dataset as ds
class DatasetGenerator:
    def __init__(self):
        self.data = np.random.sample((5, 2))
        self.label = np.random.sample((5, 1))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


dataset_generator = DatasetGenerator()
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)

for data in dataset.create_dict_iterator():
    print('{}'.format(data["data"]), '{}'.format(data["label"]))

for data in dataset.create_dict_iterator():
    print("Image shape: {}".format(data['data'].shape), ", Label: {}".format(data['label']))


#数据增强
import matplotlib.pyplot as plt

from luojianet_ms.dataset.vision import Inter
import luojianet_ms.dataset.vision.c_transforms as c_vision

DATA_DIR = './datasets/MNIST_Data/train'

mnist_dataset = ds.MnistDataset(DATA_DIR, num_samples=6, shuffle=False)

# 查看数据原图
mnist_it = mnist_dataset.create_dict_iterator()
data = next(mnist_it)
plt.figure(figsize=(3, 3))
plt.imshow(data['image'].asnumpy().squeeze(), cmap=plt.cm.gray)
plt.title(data['label'].asnumpy(), fontsize=20)
plt.show()

#增强操作，改变尺寸、随机裁剪
resize_op = c_vision.Resize(size=(200, 200), interpolation=Inter.LINEAR)
crop_op = c_vision.RandomCrop(150)
transforms_list = [resize_op, crop_op]
mnist_dataset = mnist_dataset.map(operations=transforms_list, input_columns=["image"])

mnist_dataset = mnist_dataset.create_dict_iterator()
data = next(mnist_dataset)
plt.figure(figsize=(3, 3))
plt.imshow(data['image'].asnumpy().squeeze(), cmap=plt.cm.gray)
plt.title(data['label'].asnumpy(), fontsize=20)
plt.show()