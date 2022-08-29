from geobject import get_objects
import os

image_dir = 'xx/GID5/image_RGB/'
encode_label_dir = 'xx/GID5/Annotations/'

def get_file_list(split='train'):

    id_list = os.path.join('datalist', split + '.txt')
    id_list = tuple(open(id_list, 'r'))

    image_files = [os.path.join(image_dir, id_.rstrip() + '.tif') for id_ in id_list]
    label_files = [os.path.join(encode_label_dir, id_.rstrip() + '_label.tif') for id_ in id_list]

    return image_files, label_files

# Function get_objects: get quadtree search patch info in json file.
# param[in] image_path, big_input image path.
# param[in] label_path, big_input label path.
# param[in] n_classes, number of dataset's land-cover categories.
# param[in] ignore_label, ignore label value, default is 255.
# param[in] seg_threshold, quadtree segmentation settings.
# param[in] search_block_size, basic global processing unit for big_input data. This parameter is default, do not need to change.
# param[in] max_searchsize, max output data size (max_searchsize*max_searchsize).
# param[in] json_filename, output patch info. This parameter is default, do not need to change.
# param[in] use_quadsearch, whether to use quadtree search.
# return out, json file with patch info.

train_image_path, train_label_path = get_file_list('train')
get_objects(train_image_path, train_label_path, 6, 255, 150, 4096, 512, '/patch_info_train.json', 1)

valid_image_path, valid_label_path = get_file_list('valid')
get_objects(valid_image_path, valid_label_path, 6, 255, 150, 4096, 512, '/patch_info_valid.json', 0)