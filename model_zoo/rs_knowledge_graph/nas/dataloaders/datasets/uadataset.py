"""Dataset FU generator."""
import cv2
import numpy as np

import luojianet_ms.ops as P
from luojianet_ms import Tensor
from luojianet_ms.common import dtype
import rasterio
from dataloaders.datasets.basedataset import BaseDataset


class Uadataset(BaseDataset):
    """Dataset FU generator."""
    def __init__(self,
                 root,
                 num_samples=None,
                 num_classes=12,
                 multi_scale=False,
                 flip=False,
                 ignore_label=-1,
                 base_size=512,
                 crop_size=None,
                 downsample_rate=1,
                 scale_factor=16,
                 mean=None,
                 std=None,
                 choice=None,
                 number=None):

        super(Uadataset, self).__init__(ignore_label, num_classes, base_size,
                                         crop_size, downsample_rate, scale_factor, mean, std)

        self._index = 0
        self.root = root
        if choice == 'train':
            # if number == 1:
            #     self.list_path = root + "/uadataset/mini_uad_512_train_1.lst"
            # elif number == 2:
            #     self.list_path = root + "/uadataset/mini_uad_512_train_2.lst"
            self.list_path = root + "/uadataset/uad_512_train.lst"
        elif choice == 'val':
            # self.list_path = root + "/uadataset/mini_uad_512_val.lst"
            self.list_path = root + "/uadataset/uad_512_val.lst"
        elif choice == 'test':
            self.list_path = root + "/uadataset/uad_512_test.lst"
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        list_file = open(self.list_path)
        img_list = [line.strip().split() for line in list_file]
        list_file.close()
        self.img_list = [(self.root + "/" + vector[0], self.root + "/" + vector[1]) for vector in img_list]
        self._number = len(self.img_list)

        if num_samples:
            self.files = self.files[:num_samples]

        self.all_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.valid_classes = [255, 0, 1, 2, 3, 4, 5, 6, 255, 255, 7, 8, 9, 10, 11, 255]

        self.label_mapping = dict(zip(self.all_classes, self.valid_classes))
        self.class_weights = None

    def __len__(self):
        return self._number

    def __getitem__(self, index):
        if index < self._number:
            image_path = self.img_list[index][0]
            label_path = self.img_list[index][1]
            with rasterio.open(image_path) as image:
                image = image.read().astype(np.float32).transpose(1, 2, 0)
            with rasterio.open(label_path) as label:
                label = label.read()
            if len(label.shape) == 3:
                label = label[0]
            label = self.convert_label(label)
            image, label = self.gen_sample(image, label, self.multi_scale, self.flip)
        else:
            raise StopIteration
        return image.copy(), label.copy()

    def show(self):
        """Show the total number of val data."""
        print("Total number of data vectors: ", self._number, flush=True)
        for line in self.img_list:
            print(line, flush=True)

    def convert_label(self, label, inverse=False):
        """Convert classification ids in labels."""
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def multi_scale_inference(self, model, image, scales=None, flip=False):
        """Inference using multi-scale features from dataset Cityscapes."""
        batch, _, ori_height, ori_width = image.shape
        assert batch == 1, "only supporting batchsize 1."
        image = image.asnumpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)

        final_pred = Tensor(np.zeros([1, self.num_classes, ori_height, ori_width]), dtype=dtype.float32)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = Tensor(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds.asnumpy()
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = np.zeros([1, self.num_classes, new_h, new_w]).astype(np.float32)

                count = np.zeros([1, 1, new_h, new_w]).astype(np.float32)

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = Tensor(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred.asnumpy()[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]
            preds = Tensor(preds)
            preds = P.ResizeBilinear((ori_height, ori_width))(preds)
            final_pred = P.Add()(final_pred, preds)
        return final_pred
