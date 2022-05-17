
config = dict(
    device_target = 'GPU',
    dataset_path = './data/',
    normalize = False,
    nodata_value = 0,
    in_channels = 32,
    classnum = 24,
    batch_size = 2,
    num_epochs = 100,
    weight = None,
    learning_rate = 1e-4,
    save_model_path = './model/'
)