
import logging
from skimage import io
import os
def read_txt(path):
    img_id = []
    for id in open(path):
        if len(id) > 0:
            img_id.append(id.strip())
    return img_id

class Isprs_Dataset:
    def __init__(self, img_dir, label_dir, transform, img_id_txt_path=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_id_txt_path = img_id_txt_path
        self.ids = read_txt(self.img_id_txt_path)

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_path=os.path.join(self.img_dir, idx)
        img=io.imread(img_path)[:,:,0:3]
        label_path = os.path.join(self.label_dir, idx)
        label = io.imread(label_path)
        sample = self.transform(image=img, mask=label)
        image = sample['image'].transpose((2, 0, 1))
        label = sample['mask']
        return image, label
