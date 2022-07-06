import cv2
import numpy as np
import os
from PIL import Image
from src.preprocess import image_augment, crop_input, scale_input, scale_camera


# the WHU dataset preprocessed by Jin Liu (only for training and testing)
class MVSDatasetGenerator:
    def __init__(self, data_folder, mode, view_num, normalize, args):
        self.all_data_folder = data_folder
        self.mode = mode
        self.args = args
        self.view_num = view_num
        self.normalize = normalize
        self.ndepths = args.ndepths
        self.interval_scale = args.interval_scale
        assert self.mode in ["train", "val", "test"]
        self.sample_list = self.build_list()
        self.sample_num = len(self.sample_list)

    def build_list(self):
        # Prepare all training samples
        """ generate data paths for whu dataset """
        total_sample_list = []

        # meitan/tianjin/munchen sets
        all_data_set_path = self.all_data_folder + '/index.txt'
        all_data_set = open(all_data_set_path).read().split()

        for sub_set in all_data_set:
            data_folder = self.all_data_folder + '/{}/{}'.format(sub_set, self.mode)
            sample_list = self.whu_list(data_folder, sub_set, gt_fext='.png')

            total_sample_list += sample_list

        return total_sample_list

    def get_list(self):
        return self.sample_list

    def whu_list(self, data_folder, sat_name, gt_fext='.png'):
        sample_list = []

        # image index
        train_cluster_path = data_folder + '/index.txt'
        data_cluster = open(train_cluster_path).read().split()

        # pair
        view_pair_path = data_folder + '/pair.txt'
        ref_indexs = []
        src_indexs = []
        with open(view_pair_path) as f:
            cluster_num = int(f.readline().rstrip())
            for idx in range(cluster_num):
                ref_index = int(f.readline().rstrip())
                src_index = [int(x) for x in f.readline().rstrip().split()][1:]
                ref_indexs.append(ref_index)
                src_indexs.append(src_index)

        # for each data scene
        for i in data_cluster:
            image_folder = os.path.join(data_folder, ('Images/%s' % i)).replace("\\", "/")
            cam_folder = os.path.join(data_folder, ('Cams/%s' % i)).replace("\\", "/")
            depth_folder = os.path.join(data_folder, ('Depths/%s' % i)).replace("\\", "/")

            # for each view
            for ref_ind, view_inds in zip(ref_indexs, src_indexs):  # 0-4
                image_folder2 = os.path.join(image_folder, ('%d' % ref_ind)).replace("\\", "/")
                image_files = sorted(os.listdir(image_folder2))

                view_cnts = min(self.view_num, len(view_inds) + 1)

                for j in range(0, int(np.size(image_files))):
                    paths = []
                    portion = os.path.splitext(image_files[j])
                    newcamname = portion[0] + '.txt'
                    newdepthname = portion[0] + gt_fext

                    # ref image
                    ref_image_path = os.path.join(os.path.join(image_folder, ('%d' % ref_ind)),
                                                  image_files[j]).replace("\\", "/")
                    ref_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % ref_ind)), newcamname).replace(
                        "\\", "/")

                    paths.append(ref_image_path)
                    paths.append(ref_cam_path)

                    # view images
                    for view in range(view_cnts - 1):
                        # print(image_folder)
                        view_ind = view_inds[view]  # selected view image
                        view_image_path = os.path.join(os.path.join(image_folder, ('%d' % view_ind)),
                                                       image_files[j]).replace("\\", "/")

                        view_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % view_ind)),
                                                     newcamname).replace("\\", "/")
                        paths.append(view_image_path)
                        paths.append(view_cam_path)

                    # depth path
                    depth_image_path = os.path.join(os.path.join(depth_folder, ('%d' % ref_ind)),
                                                    newdepthname).replace("\\", "/")
                    paths.append(depth_image_path)
                    sample_list.append((sat_name, view_cnts, paths))

        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def tr_read_cam_whu(self, file, ndepths, interval_scale=1):
        # read camera txt file
        cam = np.zeros((2, 4, 4), dtype=np.float32)
        extrinsics = np.zeros((4, 4), dtype=np.float32)
        pera = np.zeros((1, 13), dtype=np.float32)
        words = open(file).read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                extrinsics[i][j] = words[extrinsic_index]  # Twc

        # if cam ori is XrightYup
        O = np.eye(3, dtype=np.float32)
        O[1, 1] = -1
        O[2, 2] = -1
        R = extrinsics[0:3, 0:3]
        R2 = np.matmul(R, O)
        extrinsics[0:3, 0:3] = R2

        extrinsics = np.linalg.inv(extrinsics)  # Tcw
        cam[0, :, :] = extrinsics

        for i in range(0, 13):
            pera[0][i] = words[17 + i]

        f = pera[0][0]
        x0 = pera[0][1]  # whu
        y0 = pera[0][2]

        # K Photogrammetry system XrightYup
        cam[1][0][0] = f
        cam[1][1][1] = f
        cam[1][0][2] = x0
        cam[1][1][2] = y0
        cam[1][2][2] = 1

        # depth range
        cam[1][3][0] = np.float32(pera[0][3])  # start
        cam[1][3][1] = np.float32(pera[0][5] * interval_scale)  # interval
        cam[1][3][3] = np.float32(pera[0][4])  # end

        acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32

        if acturald > ndepths:
            scale = acturald / np.float32(ndepths)
            cam[1][3][1] = cam[1][3][1] * scale
            acturald = ndepths
        # cam[1][3][2] = acturald
        cam[1][3][2] = ndepths
        location = words[23:30]

        return cam, location

    @staticmethod
    def read_img(filename):
        img = Image.open(filename)
        return img

    @staticmethod
    def read_depth(filename):
        depimg = Image.open(filename)
        # WHU MVS dataset and Tianjin MVS dataset
        depth_image = np.array(depimg, dtype=np.float32) / 64.0

        return depth_image

    @staticmethod
    def center_image(img, mode='mean'):
        """ normalize image input """
        # attention: CasMVSNet [mean var];; CasREDNet [0-255]
        if mode == 'standard':
            np_img = np.array(img, dtype=np.float32) / 255.

        elif mode == 'mean':
            img_array = np.array(img)
            img = img_array.astype(np.float32)
            # img = img.astype(np.float32)
            var = np.var(img, axis=(0, 1), keepdims=True)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            np_img = (img - mean) / (np.sqrt(var) + 0.00000001)

        else:
            raise Exception("{}? Not implemented yet!".format(mode))

        return np_img

    def __getitem__(self, idx):
        sat_name, view_cnts, paths = self.sample_list[idx]

        ###### read input data ######
        centered_images = []
        proj_matrices = []

        # depth
        depth_image = self.read_depth(os.path.join(paths[2 * view_cnts]))

        for view in range(self.view_num):
            # Images
            if self.mode == "train":
                image = image_augment(self.read_img(paths[2 * view]))
            else:
                image = self.read_img(paths[2 * view])
            image = np.array(image)

            # Cameras
            cam, location = self.tr_read_cam_whu(paths[2 * view + 1], self.ndepths, self.interval_scale)
            location.append(str(self.args.resize_scale))

            if view == 0:
                # determine a proper scale to resize input
                scaled_image, scaled_cam, scaled_depth = scale_input(image, cam, depth_image=depth_image,
                                                                     scale=self.args.resize_scale)
                # crop to fit network
                croped_image, croped_cam, croped_depth = crop_input(scaled_image, scaled_cam, depth_image=scaled_depth,
                                                                    max_h=self.args.max_h, max_w=self.args.max_w,
                                                                    resize_scale=self.args.resize_scale)
                outimage = croped_image
                outcam = croped_cam
                outlocation = location
                depth_min = croped_cam[1][3][0]
                depth_interval = croped_cam[1][3][1]
                new_ndepths = croped_cam[1][3][2]
                depth_max = outcam[1][3][3]
            else:
                # determine a proper scale to resize input
                scaled_image, scaled_cam = scale_input(image, cam, scale=self.args.resize_scale)
                # crop to fit network
                croped_image, croped_cam = crop_input(scaled_image, scaled_cam, max_h=self.args.max_h,
                                                      max_w=self.args.max_w, resize_scale=self.args.resize_scale)

            # scale cameras for building cost volume
            scaled_cam = scale_camera(croped_cam, scale=self.args.sample_scale)

            # print(scaled_cam)

            # multiply intrinsics and extrinsics to get projection matrix
            extrinsics = scaled_cam[0, :, :]
            intrinsics = scaled_cam[1, 0:3, 0:3]
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])

            if view:
                proj_matrices.append(np.matmul(proj_mat, np.linalg.inv(proj_matrices[0])))
                # print("src_proj")
                # print(proj_mat)
                #
                # print("ref_proj")
                # print(proj_matrices[0])
                #
                # print("ref_proj_inv")
                # print(np.linalg.inv(proj_matrices[0]))

            else:
                proj_matrices.append(proj_mat)

            centered_images.append(self.center_image(croped_image, mode=self.normalize))

        # Depth
        # print(new_ndepths)
        depth_value = np.arange(
            np.float(depth_min), np.float(depth_interval * (new_ndepths - 0.5) + depth_min),
            np.float(depth_interval), dtype=np.float32)

        # print(depth_value)
        # print(depth_value.shape)
        depth_value = depth_value.reshape(-1, 1, 1).repeat(int(self.args.max_h/4), 1)
        depth_value = depth_value.repeat(int(self.args.max_w/4), 2)

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        # croped_depth = cv2.resize(croped_depth, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        mask = np.logical_and(croped_depth < depth_max, croped_depth > depth_min).astype(np.float32)

        return centered_images, proj_matrices[:, 0:3, :], croped_depth, depth_value, mask


if __name__ == "__main__":
    # some testing code, just IGNORE it
    class Args:
        def __init__(self):
            self.ndepths = 192
            self.interval_scale = 1.06
            self.resize_scale = 1
            self.max_h = 384
            self.max_w = 768
            self.sample_scale = 1.0

    dataset = MVSDatasetGenerator(
        "/mnt/gj/stereo", "train", 3, "mean", Args())

    imgs, cameras, depth, depth_int, value = dataset.__getitem__(0)

    import matplotlib.pyplot as plt
    plt.subplot(2, 2, 1)
    plt.imshow(imgs[0, 0, :, :])
    plt.subplot(2, 2, 2)
    plt.imshow(imgs[0, 1, :, :])
    plt.subplot(2, 2, 3)
    plt.imshow(imgs[0, 2, :, :])

    plt.show()
    print(imgs.shape)
    print(value)
