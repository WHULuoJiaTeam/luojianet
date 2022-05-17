import numpy as np

def label2rgb_hanchuan(label):
    """"""
    label2color_dict = {
        0: [0, 0, 0],
        1: [176, 48, 96],
        2: [0, 255, 255],
        3: [255, 0, 255],
        4: [160, 32, 240],
        5: [127, 255, 212],
        6: [127, 255, 0],
        7: [0, 205, 0],
        8: [0, 255, 0],
        9: [0, 139, 0],
        10: [255, 0, 0],
        11: [216, 191, 216],
        12: [255, 127, 80],
        13: [160, 82, 45],
        14: [255, 255, 255],
        15: [218, 112, 214],
        16: [0, 0, 255]
    }
    # np.array([[255, 255, 255], [255, 0, 0],
    #           [255, 255, 0], [0, 255, 0], [0, 255, 255],
    #           [0, 0, 255]])
    # visualize
    visual_anno = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for class_index in label2color_dict:
        visual_anno[label == class_index, :]=label2color_dict[class_index]
    return visual_anno

def label2rgb_longkou(label):
    """"""
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [238, 154, 0],
        3: [255, 255, 0],
        4: [0, 255, 0],
        5: [0, 255, 255],
        6: [0, 139, 139],
        7: [0, 0, 255],
        8: [255, 255, 255],
        9: [160, 32, 240]
    }
    # np.array([[255, 255, 255], [255, 0, 0],
    #           [255, 255, 0], [0, 255, 0], [0, 255, 255],
    #           [0, 0, 255]])
    # visualize
    visual_anno = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for class_index in label2color_dict:
        visual_anno[label == class_index, :]=label2color_dict[class_index]
    return visual_anno

def label2rgb_honghu(label):
    """"""
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [255, 255, 255],
        3: [176, 48, 96],
        4: [255, 255, 0],
        5: [255, 127, 80],
        6: [0, 255, 0],
        7: [0, 205, 0],
        8: [0, 139, 0],
        9: [127, 255, 212],
        10: [160, 32, 240],
        11: [216, 191, 216],
        12: [255, 255, 255],
        13: [0, 0, 255],
        14: [0, 0, 139],
        15: [218, 112, 214],
        16: [160, 82, 45],
        17: [0, 255, 255],
        18: [255, 165, 0],
        19: [127, 255, 0],
        20: [139, 139, 0],
        21: [0, 139, 139],
        22: [205, 181, 205],
        23: [238, 154, 0]
    }
    # np.array([[255, 255, 255], [255, 0, 0],
    #           [255, 255, 0], [0, 255, 0], [0, 255, 255],
    #           [0, 0, 255]])
    # visualize
    visual_anno = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for class_index in label2color_dict:
        visual_anno[label == class_index, :]=label2color_dict[class_index]
    return visual_anno
