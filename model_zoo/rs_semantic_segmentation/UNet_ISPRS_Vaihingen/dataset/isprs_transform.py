import albumentations as A

def train_transform():
    transform = A.Compose([
        A.ToFloat(max_value=1.0),
        A.OneOf([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5)
        ]),
        A.Normalize(mean=(0.491, 0.482, 0.447),
                    std=(0.247, 0.243, 0.262), max_pixel_value=255.0),
    ])
    return transform
def val_transform():
    transform = A.Compose([
        A.ToFloat(max_value=1.0),
        A.Normalize(mean=(0.491, 0.482, 0.447),
                    std=(0.247, 0.243, 0.262), max_pixel_value=255.0),
    ])
    return transform