
config = dict(
    device_target = 'GPU', # 设备类型，CPU或者GPU
    dataset_path = './data/', # 数据集根目录，如组织格式示例中的data文件夹所在位置
    normalize = False, # 是否对影像进行归一化，False或True，若为True，则逐波段进行标准差归一化
    nodata_value = 0, # 标签中的Nodata值（即不作为样本的像素值）
    in_channels = 32, # 输入通道数（即影像波段数）
    classnum = 24, # 类别数量
    batch_size = 2, # 训练时的batchsize
    num_epochs = 100, # 训练迭代次数
    weight = None, # 是否在损失函数中对各类别加权，默认为不加权（None），若需要加权，则给出一个各类别权重的list
    learning_rate = 1e-4, # 训练学习率
    save_model_path = './model/' # 训练模型文件保存路径
)