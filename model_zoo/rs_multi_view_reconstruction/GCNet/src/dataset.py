
import numpy as np
from src.data_io import imread
from src.data_io import read_pfm


class DatasetGenerator:
    def __init__(self, list_file, crop_h, crop_w, dataset):
        self.list_info = self.read_list(list_file)
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.dataset = dataset

    @staticmethod
    def read_list(list_path):
        with open(list_path, "r") as f:
            data = f.read().splitlines()

        data = [d.split(",") for d in data]

        return data

    @staticmethod
    def random_position(crop_h, crop_w, h, w):
        if crop_w != w and crop_h != h:
            y = np.random.randint(0, h - crop_h - 1)
            x = np.random.randint(0, w - crop_w - 1)

            return x, y
        else:
            return 0, 0

    @staticmethod
    def normalize_image(img):
        img = img.astype(np.float32)
        img /= 255.0
        return img

    def __getitem__(self, index):
        left_img_path, right_img_path, gt_path = self.list_info[index]

        left_img = imread(left_img_path)
        right_img = imread(right_img_path)

        if self.dataset == "kitti":
            disp = imread(gt_path).astype(np.float32)
            disp /= 256.0
        elif self.dataset == "sceneflow":
            disp = read_pfm(gt_path).astype(np.float32)
            disp[disp < 0] = 0
            disp[disp >= 192] = 0
        elif self.dataset == "whu":
            disp = imread(gt_path).astype(np.float32)
            disp /= 256.0
        else:
            disp = None
            Exception("Not supported now")

        x, y = self.random_position(self.crop_h, self.crop_w, left_img.shape[0], left_img.shape[1])

        # print("random position: {} {}".format(x, y))
        left_img = left_img[y:y + self.crop_h, x:x + self.crop_w, :]
        right_img = right_img[y:y + self.crop_h, x: x + self.crop_w, :]
        disp = disp[y:y + self.crop_h, x:x + self.crop_w]

        left_img = self.normalize_image(left_img).transpose(2, 0, 1)
        right_img = self.normalize_image(right_img).transpose(2, 0, 1)

        images = []
        images.append(left_img)
        images.append(right_img)
        images = np.stack(images, axis=0)

        return (images.astype(np.float32), disp.astype(np.float32))

    def __len__(self):
        return len(self.list_info)


if __name__ == "__main__":
    data = DatasetGenerator("list/sceneflow_validation.txt")

    d_min = 0
    d_max = 0
    for d in range(data.__len__()):
        imgs, gt = data.__getitem__(d)
        print(d, np.min(gt), np.max(gt))
        if np.min(gt) < d_min:
            d_min = np.min(gt)
        if np.max(gt) > d_max:
            d_max = np.max(gt)

    print(d_min, d_max)

    # print(gt)
    # plt.imshow(gt)
    # plt.show()

