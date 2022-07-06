import numpy as np

class SegEvaluator:
    def __init__(self, class_num=4):
        if class_num == 1:
            class_num = 2
        self.num_class = class_num
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def pixel_oa(self):
        pixel_accuracy = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return pixel_accuracy

    def pixel_aa(self):  # AA
        pixel_aa = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        pixel_aa = np.nanmean(pixel_aa)
        return pixel_aa

    def mean_iou(self):
        miou = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        miou = np.nanmean(miou)
        return miou

    def class_iou(self, class_index):
        iou_per_class = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return iou_per_class[class_index]

    def frequency_weighted_iou(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwiou

    def kappa(self):
        pe_rows = np.sum(self.confusion_matrix, axis=0)
        pe_cols = np.sum(self.confusion_matrix, axis=1)
        sum_total = np.sum(self.confusion_matrix)
        pe = np.dot(pe_rows, pe_cols) / (sum_total ** 2)
        po = self.pixel_oa()
        return (po - pe) / (1 - pe)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def one_class_classreport(self,class_index):
        TP=self.confusion_matrix[class_index,class_index]#[TP,FN]
        FN=self.confusion_matrix[class_index,:].sum()-TP #[FP,TN]
        FP=self.confusion_matrix[:,class_index].sum()-TP
        p=TP/(TP+FP)
        r=TP/(TP+FN)
        f1=2*p*r/(p+r)
        return p,r,f1

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
