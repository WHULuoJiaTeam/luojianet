import numpy as np
from scipy.io import loadmat
import luojianet_ms as ms
import luojianet_ms.dataset as ds

def minibatch_sample(gt_mask, minibatch_size):
    rs = np.random.RandomState(5)

    cls_list = np.unique(gt_mask)
    cls_list = np.delete(cls_list, 0)
    # print(cls_list)
    idx_dict_per_class = dict()
    for cls in cls_list:

        train_idx_per_class = np.where(gt_mask == cls, gt_mask, np.zeros_like(gt_mask))
        idx = np.where(train_idx_per_class.ravel() == cls)[0]

        rs.shuffle(idx)

        idx_dict_per_class[cls] = idx

    train_idx_list = []
    count = 0
    while True:
        train_idx = np.zeros_like(gt_mask).ravel()
        for cls, idx in idx_dict_per_class.items():
            left = count * minibatch_size
            if left >= len(idx):
                continue
            # remain last batch though the real size is smaller than minibatch_size
            right = min((count + 1) * minibatch_size, len(idx))
            fetch_idx = idx[left:right]
            train_idx[fetch_idx] = cls
        count += 1
        if train_idx.sum() == 0:
            return train_idx_list
        train_idx_list.append(train_idx.reshape(gt_mask.shape))


def image_pad(image, encoder_size):
    image_shape = image.shape
    image_shape = np.array(image_shape)
    pad_shape = image_shape

    pad_shape[1] = int(np.ceil(image_shape[1] / encoder_size) * encoder_size)
    pad_shape[2] = int(np.ceil(image_shape[2] / encoder_size) * encoder_size)

    out = np.zeros((pad_shape[0], pad_shape[1], pad_shape[2]))
    out[:, 0:image.shape[1], 0:image.shape[2]] = image

    return out


class WHU_Hi_dataloader():
    def __init__(self, config):
        self.train_gt = loadmat(config['train_gt_dir'])[config['train_gt_name']]
        self.data = loadmat(config['train_data_dir'])[config['train_data_name']]

        im_cmean = self.data.reshape((-1, self.data.shape[-1])).mean(axis=0)
        im_cstd = self.data.reshape((-1, self.data.shape[-1])).std(axis=0)
        self.data = (self.data - im_cmean) / im_cstd

        self.data = self.data.transpose(2, 0, 1)

        self.data = image_pad(self.data, config['encoder_size'])
        self.data = self.data.reshape(1, self.data.shape[0], self.data.shape[1], self.data.shape[2])
        self.data = ms.Tensor.from_numpy(self.data.astype(np.float32))

        self.train_gt = self.train_gt.reshape(1, self.train_gt.shape[0], self.train_gt.shape[1])
        self.train_gt = image_pad(self.train_gt, config['encoder_size'])

        self.train_gt_idx = minibatch_sample(self.train_gt, 5)

    def __getitem__(self, index):
        temp = self.train_gt_idx[index].astype(np.float32) - 1

        return self.data, ms.Tensor.from_numpy(temp)


    def __len__(self):
        return len(self.train_gt_idx)


class WHU_Hi_test():
    def __init__(self, config):
        self.test_gt = loadmat(config['test_gt_dir'])[config['test_gt_name']]
        self.data = loadmat(config['test_data_dir'])[config['test_data_name']]

        im_cmean = self.data.reshape((-1, self.data.shape[-1])).mean(axis=0)
        im_cstd = self.data.reshape((-1, self.data.shape[-1])).std(axis=0)
        self.data = (self.data - im_cmean) / im_cstd

        self.data = self.data.transpose(2, 0, 1)

        self.data = image_pad(self.data, config['encoder_size'])
        self.data = self.data.reshape(1, self.data.shape[0], self.data.shape[1], self.data.shape[2])
        self.data = ms.Tensor.from_numpy(self.data.astype(np.float32))

        self.test_gt = self.test_gt.reshape(1, self.test_gt.shape[0], self.test_gt.shape[1])
        self.test_gt = image_pad(self.test_gt, config['encoder_size'])

    def return_data(self):
        return self.data, ms.Tensor.from_numpy(self.test_gt)


class WHU_Hi_map():
    def __init__(self, config):
        self.data = loadmat(config['alldata_dir'])[config['alldata_name']]

        im_cmean = self.data.reshape((-1, self.data.shape[-1])).mean(axis=0)
        im_cstd = self.data.reshape((-1, self.data.shape[-1])).std(axis=0)
        self.data = (self.data - im_cmean) / im_cstd

        self.data = self.data.transpose(2, 0, 1)

        self.data = image_pad(self.data, config['encoder_size'])
        self.data = self.data.reshape(1, self.data.shape[0], self.data.shape[1], self.data.shape[2])
        self.data = ms.Tensor.from_numpy(self.data.astype(np.float))


    def return_data(self):
        return self.data


# train_data = WHU_Hi_dataloader(S3ANet_HH_config['dataset']['params'])
# train_data = ds.GeneratorDataset(train_data, ["data", "label"])
# model = S3ANET(S3ANet_HH_config['model']['params'], train=True)
# for data in train_data.create_dict_iterator():
#     b = model(data['data'], data['label'])
#     print('{}'.format(data['data']), '{}'.format(data['label']))