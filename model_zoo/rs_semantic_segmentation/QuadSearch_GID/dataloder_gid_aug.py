import numpy as np
from osgeo import gdal, gdalconst
import os
import json
from tqdm import tqdm
from PIL import Image, ImageEnhance
import luojianet_ms.dataset.vision.c_transforms as c_vision
import luojianet_ms.dataset.vision.py_transforms as p_vision
from luojianet_ms.dataset.transforms import c_transforms
from luojianet_ms.dataset.transforms import py_transforms


class Dataset_RealUD:
    def __init__(self, encode_label_dir, color_label_dir, json_path, blocksize=512, return_save_info=False, train=False):
        self.encode_label_dir = encode_label_dir
        self.color_label_dir = color_label_dir
        self.json_path = json_path
        self.return_save_info = return_save_info
        with open(json_path, 'r') as load_f:
            self.dict_dataset = json.load(load_f)
        self.length = len(self.dict_dataset)

        self.train = train

        self.imageAugment = c_transforms.Compose([
            c_vision.RandomHorizontalFlip(prob=0.5),
            c_vision.RandomVerticalFlip(prob=0.5),
        ])

        self.img_transform = py_transforms.Compose([
                                p_vision.ToTensor(),
                                p_vision.Normalize([0.3309, 0.3473, 0.3247], [0.2560, 0.2512, 0.2468])
                            ])


        self.blocksize = blocksize


    def __getitem__(self, index):
        Dict_i = self.dict_dataset[str(index)]
        image_path = Dict_i['imagePath']
        label_path = Dict_i['labelPath'].replace(self.encode_label_dir, self.color_label_dir)  # color label
        # label_path = Dict_i['labelPath'].replace('/media/xx/PortableSSD/GID5/label/', '/media/xx/PortableSSD/GID5/label_5classes/')  # color label
        # image_path = Dict_i['imagePath'].replace('/data01/SegDataset/GID5/image_RGB/', '/media/xx/PortableSSD/GID5/image/')
        # label_path = Dict_i['labelPath'].replace('/data01/SegDataset/GID5/Annotations/', '/media/xx/PortableSSD/GID5/label_5classes/')
        # print(image_path)
        # print(label_path)

        currentImgdata = gdal.Open(image_path, gdalconst.GA_ReadOnly)
        currentLabeldata = gdal.Open(label_path, gdalconst.GA_ReadOnly)

        imgdata_band1 = currentImgdata.GetRasterBand(1)
        imgdata_band2 = currentImgdata.GetRasterBand(2)
        imgdata_band3 = currentImgdata.GetRasterBand(3)
        imagepatch_1 = imgdata_band1.ReadAsArray(Dict_i['x'], Dict_i['y'], Dict_i['block_x'], Dict_i['block_y']).astype('uint8')
        imagepatch_2 = imgdata_band2.ReadAsArray(Dict_i['x'], Dict_i['y'], Dict_i['block_x'], Dict_i['block_y']).astype('uint8')
        imagepatch_3 = imgdata_band3.ReadAsArray(Dict_i['x'], Dict_i['y'], Dict_i['block_x'], Dict_i['block_y']).astype('uint8')
        imagepatch_i = np.stack([imagepatch_1, imagepatch_2, imagepatch_3], axis=2)

        labeldata_band1 = currentLabeldata.GetRasterBand(1)
        labeldata_band2 = currentLabeldata.GetRasterBand(2)
        labeldata_band3 = currentLabeldata.GetRasterBand(3)
        labelpatch_1 = labeldata_band1.ReadAsArray(Dict_i['x'], Dict_i['y'], Dict_i['block_x'], Dict_i['block_y']).astype('uint8')
        labelpatch_2 = labeldata_band2.ReadAsArray(Dict_i['x'], Dict_i['y'], Dict_i['block_x'], Dict_i['block_y']).astype('uint8')
        labelpatch_3 = labeldata_band3.ReadAsArray(Dict_i['x'], Dict_i['y'], Dict_i['block_x'], Dict_i['block_y']).astype('uint8')
        labelpatch_i = np.zeros_like(labelpatch_1).astype('uint8')
        labelpatch_i = np.where((labelpatch_1 == 255) * (labelpatch_2 == 0) * (labelpatch_3 == 0), 1, labelpatch_i)
        labelpatch_i = np.where((labelpatch_1 == 0) * (labelpatch_2 == 255) * (labelpatch_3 == 0), 2, labelpatch_i)
        labelpatch_i = np.where((labelpatch_1 == 0) * (labelpatch_2 == 255) * (labelpatch_3 == 255), 3, labelpatch_i)
        labelpatch_i = np.where((labelpatch_1 == 255) * (labelpatch_2 == 255) * (labelpatch_3 == 0), 4, labelpatch_i)
        labelpatch_i = np.where((labelpatch_1 == 0) * (labelpatch_2 == 0) * (labelpatch_3 == 255), 5, labelpatch_i)
        # print(torch.max(torch.tensor(labelpatch_i)))
        # labelpatch_i = np.where((labelpatch_1 == 0) * (labelpatch_2 == 0) * (labelpatch_3 == 0), 6, labelpatch_i).astype('uint8')
        # labelpatch_t = np.stack([labelpatch_1, labelpatch_2, labelpatch_3], axis=2)

        # print(imagepatch_i, labelpatch_i)
        if Dict_i['block_x'] < self.blocksize or Dict_i['block_y'] < self.blocksize:
            imagepatch_i = Image.fromarray(imagepatch_i)
            labelpatch_i = Image.fromarray(labelpatch_i)
            imagepatch_i = np.array(
                p_vision.Pad((0, 0, self.blocksize - Dict_i['block_x'], self.blocksize - Dict_i['block_y']),
                             fill_value=0)(imagepatch_i))
            labelpatch_i = np.array(
                p_vision.Pad((0, 0, self.blocksize - Dict_i['block_x'], self.blocksize - Dict_i['block_y']),
                             fill_value=6)(labelpatch_i))
        if self.train:
            imagepatch_i = np.uint8(imagepatch_i)
            img_label = np.concatenate([imagepatch_i, labelpatch_i[:, :, np.newaxis]], axis=2)

            img_label = Image.fromarray(img_label)
            img_label = self.imageAugment(img_label)
            img_label = np.array(img_label)
            image = img_label[:, :, :3]
            label = img_label[:, :, 3]

            random_f = np.random.randint(1, 15)
            if random_f <= 3:
                label = label + 1
                random_angle = np.random.randint(1, 360)
                image = np.array(Image.fromarray(image).rotate(random_angle, Image.BICUBIC))
                label = np.array(Image.fromarray(label).rotate(random_angle, Image.NEAREST))
                label[label == 0] = 7
                label = label - 1
            if random_f >= 2 and random_f <= 5:
                image = Image.fromarray(image)
                random_factor = np.random.randint(0, 31) / 10.
                image = ImageEnhance.Color(image).enhance(random_factor)
                random_factor = np.random.randint(10, 21) / 10.
                image = ImageEnhance.Brightness(image).enhance(random_factor)
                random_factor = np.random.randint(10, 21) / 10.
                image = ImageEnhance.Contrast(image).enhance(random_factor)
                random_factor = np.random.randint(0, 31) / 10.
                image = ImageEnhance.Sharpness(image).enhance(random_factor)
                image = np.array(image)
        else:
            imagepatch_i = np.uint8(imagepatch_i)
            image = imagepatch_i
            label = labelpatch_i

        # image, label = imagepatch_i, labelpatch_i

        image = self.img_transform(image)

        if self.return_save_info:
            return image, label, np.array(index)

        return image, label

    def __len__(self):
        return self.length

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from luojianet_ms import dataset
    dataset_generator = Dataset_RealUD(json_path=r'/media/vhr/0D7A09740D7A0974/zz/GID/LCC5C/LCC5C_b512_woOverlap.json',
        return_save_info=True,train=True)
    dataset = dataset.GeneratorDataset(dataset_generator, ["data", "label", "index"], shuffle=False)
    dataset = dataset.batch(8)

    for data in tqdm(dataset.create_dict_iterator()):
        pass
        print(data['data'][:,0,:,:,:].shape, data['label'].shape, data['index'])
        # print('{}'.format(data["data"]), '{}'.format(data["label"]))









