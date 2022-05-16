#!/usr/bin/env python


import numpy as np
from luojianet_ms import Tensor
from luojianet_ms import dtype as mstype
import luojianet_ms.nn as nn
from luojianet_ms.ops import operations as P
from luojianet_ms.ops import composite as C
from luojianet_ms.ops import constexpr
import luojianet_ms.ops.functional as F


class HomoWarp(nn.Module):
    '''STN'''

    def __init__(self, H, W):
        super(HomoWarp, self).__init__()
        # batch_size = 1
        x = np.linspace(0, W - 1, W)
        y = np.linspace(0, H - 1, H)
        x_t, y_t = np.meshgrid(x, y)
        x_t = Tensor(x_t, mstype.float32)
        y_t = Tensor(y_t, mstype.float32)
        expand_dims = P.ExpandDims()
        x_t = expand_dims(x_t, 0)
        y_t = expand_dims(y_t, 0)
        flatten = P.Flatten()
        x_t_flat = flatten(x_t)
        y_t_flat = flatten(y_t)
        oneslike = P.OnesLike()
        ones = oneslike(x_t_flat)
        concat = P.Concat()
        sampling_grid = concat((x_t_flat, y_t_flat, ones))
        self.sampling_grid = expand_dims(sampling_grid, 0)  # (1, 3, D*H*W)
        c = np.linspace(0, 31, 32)
        self.channel = Tensor(c, mstype.float32).view(1, 1, 1, 1, -1)

        batch_size = 1
        batch_idx = np.arange(batch_size)
        batch_idx = batch_idx.reshape((batch_size, 1, 1, 1))
        self.batch_idx = Tensor(batch_idx, mstype.float32)
        self.zero = Tensor(np.zeros([]), mstype.float32)

    def get_pixel_value(self, img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.

        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*D*H*W,)
        - y: flattened tensor of shape (B*D*H*W,)

        Returns
        -------
        - output: tensor of shape (B, D, H, W, C)
        """
        shape = P.Shape()
        img_shape = shape(x)
        batch_size = img_shape[0]
        D = img_shape[1]
        H = img_shape[2]
        W = img_shape[3]
        img[:, 0, :, :] = self.zero
        img[:, H - 1, :, :] = self.zero
        img[:, :, 0, :] = self.zero
        img[:, :, W - 1, :] = self.zero

        tile = P.Tile()
        batch_idx = P.Slice()(self.batch_idx, (0, 0, 0, 0), (batch_size, 1, 1, 1))
        b = tile(batch_idx, (1, D, H, W))

        expand_dims = P.ExpandDims()
        b = expand_dims(b, 4)
        x = expand_dims(x, 4)
        y = expand_dims(y, 4)

        concat = P.Concat(4)
        indices = concat((b, y, x))

        cast = P.Cast()
        indices = cast(indices, mstype.int32)
        gather_nd = P.GatherNd()

        return cast(gather_nd(img, indices), mstype.float32)

    def homo_warp(self, height, width, proj_mat, depth_values):
        """`
        This function returns a sampling grid, which when
        used with the bilinear sampler on the input feature
        map, will create an output feature map that is an
        affine transformation [1] of the input feature map.

        zero = Tensor(np.zeros([]), mstype.float32)
        Input
        -----
        - height: desired height of grid/output. Used
          to downsample or upsample.

        - width: desired width of grid/output. Used
          to downsample or upsample.

        - proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"

        - depth_values: (B, D, H, W)


        Returns
        -------
        - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
          The 2nd dimension has 2 components: (x, y) which are the
          sampling points of the original image for each point in the
          target image.

        Note
        ----
        [1]: the affine transformation allows cropping, translation,
             and isotropic scaling.
        """
        shape = P.Shape()
        B = shape(depth_values)[0]
        D = shape(depth_values)[1]
        H = height
        W = width

        R = proj_mat[:, :, :3]  # (B, 3, 3)
        T = proj_mat[:, :, 3:]  # (B, 3, 1)

        cast = P.Cast()
        depth_values = cast(depth_values, mstype.float32)

        # transform the sampling grid - batch multiply
        matmul = P.BatchMatMul()
        tile = P.Tile()
        ref_grid_d = tile(self.sampling_grid, (B, 1, 1))  # (B, 3, H*W)
        cast = P.Cast()
        ref_grid_d = cast(ref_grid_d, mstype.float32)

        # repeat_elements has problem, can not be used
        ref_grid_d = P.Tile()(ref_grid_d, (1, 1, D))

        src_grid_d = matmul(R, ref_grid_d) + T / depth_values.view(B, 1, D * H * W)

        # project negative depth pixels to somewhere outside the image
        negative_depth_mask = src_grid_d[:, 2:] <= 1e-7
        src_grid_d[:, 0:1][negative_depth_mask] = W
        src_grid_d[:, 1:2][negative_depth_mask] = H
        src_grid_d[:, 2:3][negative_depth_mask] = 1

        src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:]  # divide by depth (B, 2, D*H*W)

        reshape = P.Reshape()
        src_grid = reshape(src_grid, (B, 2, D, H, W))

        return src_grid

    def bilinear_sampler(self, img, x, y):
        """
        Performs bilinear sampling of the input images according to the
        normalized coordinates provided by the sampling grid. Note that
        the sampling is done identically for each channel of the input.

        To test if the function works properly, output image should be
        identical to input image when theta is initialized to identity
        transform.

        Input
        -----
        - img: batch of images in (B, H, W, C) layout.
        - grid: x, y which is the output of affine_grid_generator.

        Returns
        -------
        - out: interpolated images according to grids. Same size as grid.
        """
        shape = P.Shape()
        H = shape(img)[1]
        W = shape(img)[2]
        cast = P.Cast()
        max_y = cast(H - 1, mstype.float32)
        max_x = cast(W - 1, mstype.float32)
        zero = self.zero

        # grab 4 nearest corner points for each (x_i, y_i)
        floor = P.Floor()
        x0 = floor(x)
        x1 = x0 + 1
        y0 = floor(y)
        y1 = y0 + 1

        # clip to range [0, H-1/W-1] to not violate img boundaries
        x0 = C.clip_by_value(x0, zero, max_x)
        x1 = C.clip_by_value(x1, zero, max_x)
        y0 = C.clip_by_value(y0, zero, max_y)
        y1 = C.clip_by_value(y1, zero, max_y)

        # get pixel value at corner coords
        Ia = self.get_pixel_value(img, x0, y0)
        Ib = self.get_pixel_value(img, x0, y1)
        Ic = self.get_pixel_value(img, x1, y0)
        Id = self.get_pixel_value(img, x1, y1)

        # recast as float for delta calculation
        x0 = cast(x0, mstype.float32)
        x1 = cast(x1, mstype.float32)
        y0 = cast(y0, mstype.float32)
        y1 = cast(y1, mstype.float32)

        # calculate deltas
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # add dimension for addition
        expand_dims = P.ExpandDims()
        wa = expand_dims(wa, 4)
        wb = expand_dims(wb, 4)
        wc = expand_dims(wc, 4)
        wd = expand_dims(wd, 4)

        # compute output
        add_n = P.AddN()
        out = add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return out

    def call(self, input_fmap, proj_mat, depth_values, out_dims=None, **kwargs):
        """
        Spatial Transformer Network layer implementation as described in [1].

        The layer is composed of 3 elements:

        - localization_net: takes the original image as input and outputs
          the parameters of the affine transformation that should be applied
          to the input image.

        - affine_grid_generator: generates a grid of (x,y) coordinates that
          correspond to a set of points where the input should be sampled
          to produce the transformed output.

        - bilinear_sampler: takes as input the original image and the grid
          and produces the output image using bilinear interpolation.

        Input
        -----
        - input_fmap: output of the previous layer. Can be input if spatial
          transformer layer is at the beginning of architecture. Should be
          a tensor of shape (B, H, W, C)->(B, C, H, W).

        - theta: affine transform tensor of shape (B, 6). Permits cropping,
          translation and isotropic scaling. Initialize to identity matrix.
          It is the output of the localization network.

        - proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"

        - depth_values: (B, D, H, W)

        Returns
        -------
        - out_fmap: transformed input feature map. Tensor of size (B, C, H, W)-->(B, H, W, C).
        - out: (B, C, D, H, W)


        Notes
         -----
        [1]: 'Spatial Transformer Networks', Jaderberg et. al,
             (https://arxiv.org/abs/1506.02025)
        """
        # print("proj_mat")
        # print(proj_mat)
        # print("depth_values")
        # print(depth_values)

        # grab input dimensions
        trans = P.Transpose()
        input_fmap = trans(input_fmap, (0, 2, 3, 1))
        shape = P.Shape()
        input_size = shape(input_fmap)
        H = input_size[1]
        W = input_size[2]

        # generate grids of same size or upsample/downsample if specified
        if out_dims:
            out_H = out_dims[0]
            out_W = out_dims[1]
            batch_grids = self.homo_warp(out_H, out_W, proj_mat, depth_values)
        else:
            batch_grids = self.homo_warp(H, W, proj_mat, depth_values)

        # turn off the gradient of the grid
        # batch_grids = F.stop_gradient(batch_grids)
        x_s, y_s = P.Split(1, 2)(batch_grids)
        squeeze = P.Squeeze(1)
        x_s = squeeze(x_s)
        y_s = squeeze(y_s)
        out_fmap = self.bilinear_sampler(input_fmap, x_s, y_s)

        return trans(out_fmap, (0, 4, 1, 2, 3))

#
# def get_homographies(left_cam, right_cam, depth):
#     slice = ops.Slice()
#     cast = ops.Cast()
#     matrix_inverse = ops.MatrixInverse(adjoint=False)
#     transpose = ops.Transpose()
#     tile = ops.Tile()
#     reshape = ops.Reshape()
#     eye = ops.Eye()
#
#     # cameras (K, R, t)
#     R_left = slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
#     R_right = slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
#     t_left = slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
#     t_right = slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
#     K_left = slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
#     K_right = slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
#
#     print(K_left)
#
#     # preparation
#     # num_depth = depth.shape[0]
#     #
#     # K_left_inv = matrix_inverse(K_left.squeeze(1))
#     # R_left_trans = transpose(R_left.squeeze(1), (0, 2, 1))
#     # R_right_trans = transpose(R_right.squeeze(1), (0, 2, 1))
#     #
#     # fronto_direction = slice(R_left.squeeze(1), [0, 2, 0], [-1, 1, 3])  # (B, D, 1, 3)
#     #
#     # c_left = -ops.matmul(R_left_trans, t_left.squeeze(1))
#     # c_right = -ops.matmul(R_right_trans, t_right.squeeze(1))  # (B, D, 3, 1)
#     # c_relative = c_right - c_left
#     #
#     # # compute
#     # batch_size = R_left.shape[0]
#     # temp_vec = ops.matmul(c_relative, fronto_direction)
#     # depth_mat = tile(reshape(depth, [batch_size, num_depth, 1, 1]), [1, 1, 3, 3])
#     #
#     # temp_vec = tile(temp_vec.expand_dims(axis=1), [1, num_depth, 1, 1])
#     #
#     # middle_mat0 = eye(3, (batch_size, num_depth)) - temp_vec / depth_mat
#     # middle_mat1 = tile(ops.matmul(R_left_trans, K_left_inv).expand_dims(axis=1), [1, num_depth, 1, 1])
#     # middle_mat2 = ops.matmul(middle_mat0, middle_mat1)
#     #
#     # homographies = ops.matmul(tile(K_right, [1, num_depth, 1, 1]),
#     #                           ops.matmul(tile(R_right, [1, num_depth, 1, 1]), middle_mat2))
#     #
#     # return homographies
#
#     return depth
#
#
# def get_pixel_grids(height, width):
#     reshape = ops.Reshape()
#     meshgrid = ops.Meshgrid(indexing="xy")
#     oneslike = ops.OnesLike()
#     concat_axis_0 = ops.Concat(axis=0)
#
#     # texture coordinate
#     x_linspace = nn.Range(0.5, width - 0.5)
#     y_linspace = nn.Range(0.5, height - 0.5)
#     x_coordinates, y_coordinates = meshgrid((x_linspace, y_linspace))
#     x_coordinates = reshape(x_coordinates, [-1])
#     y_coordinates = reshape(y_coordinates, [-1])
#     ones = oneslike(x_coordinates)
#     indices_grid = concat_axis_0([x_coordinates, y_coordinates, ones])
#
#     return indices_grid
#
#
# def repeat_int(x, num_repeats):
#     ones_op = ops.Ones()
#     reshape = ops.Reshape()
#
#     ones = ones_op((1, num_repeats), mstype.int32)
#     x = reshape(x, (-1, 1))
#     x = ops.matmul(x, ones)
#     return reshape(x, [-1])
#
#
# def repeat_float(x, num_repeats):
#     ones_op = ops.Ones()
#     reshape = ops.Reshape()
#
#     ones = ones_op((1, num_repeats), mstype.float32)
#     x = reshape(x, (-1, 1))
#     x = ops.matmul(x, ones)
#     return reshape(x, [-1])
#
#
# def interpolate(image, x, y):
#     stack_axis_1 = ops.Stack(axis=1)
#     cast = ops.Cast()
#     floor = ops.Floor()
#     gather_nd = ops.GatherNd()
#     expand_dims = ops.ExpandDims()
#
#     batch_size = image.shape[0]
#     height =image.shape[1]
#     width = image.shape[2]
#
#     # image coordinate to pixel coordinate
#     x = x - 0.5
#     y = y - 0.5
#     x0 = cast(floor(x), mstype.int32)
#     x1 = x0 + 1
#     y0 = cast(floor(y), mstype.int32)
#     y1 = y0 + 1
#     max_y = cast(height - 1, mstype.int32)
#     max_x = cast(width - 1, mstype.int32)
#
#     min_value = Tensor(0, mstype.int32)
#     x0 = ops.clip_by_value(x0, min_value, max_x)
#     x1 = ops.clip_by_value(x1, min_value, max_x)
#     y0 = ops.clip_by_value(y0, min_value, max_y)
#     y1 = ops.clip_by_value(y1, min_value, max_y)
#     b = repeat_int(nn.Range(batch_size), height * width)
#
#     indices_a = stack_axis_1([b, y0, x0])
#     indices_b = stack_axis_1([b, y0, x1])
#     indices_c = stack_axis_1([b, y1, x0])
#     indices_d = stack_axis_1([b, y1, x1])
#
#     pixel_values_a = gather_nd(image, indices_a)
#     pixel_values_b = gather_nd(image, indices_b)
#     pixel_values_c = gather_nd(image, indices_c)
#     pixel_values_d = gather_nd(image, indices_d)
#
#     x0 = cast(x0, mstype.float32)
#     x1 = cast(x1, mstype.float32)
#     y0 = cast(y0, mstype.float32)
#     y1 = cast(y1, mstype.float32)
#
#     area_a = expand_dims(((y1 - y) * (x1 - x)), 1)
#     area_b = expand_dims(((y1 - y) * (x - x0)), 1)
#     area_c = expand_dims(((y - y0) * (x1 - x)), 1)
#     area_d = expand_dims(((y - y0) * (x - x0)), 1)
#     output = area_a * pixel_values_a + area_b * pixel_values_b + area_c * pixel_values_c + area_d * pixel_values_d
#
#     return output
#
#
# def homography_warping(input_image, homography):
#     """
#     :param input_image: with shape B H W C
#     :param homography: with shape B H W C
#     :return:
#     """
#     slice = ops.Slice()
#     expand_dims = ops.ExpandDims()
#     tile = ops.Tile()
#     reshape = ops.Reshape()
#     cast = ops.Cast()
#     equal = ops.Equal()
#     div = ops.Div()
#     unstack_axis_1 = ops.Unstack(axis=1)
#
#     batch_size = input_image.shape[0]
#     height = input_image.shape[1]
#     width = input_image.shape[2]
#
#     # turn homography to affine_mat of size (B, 2, 3) and div_mat of size (B, 1, 3)
#     affine_mat = slice(homography, [0, 0, 0], [-1, 2, 3])
#     div_mat = slice(homography, [0, 2, 0], [-1, 1, 3])
#
#     # generate pixel grids of size (B, 3, (W+1) x (H+1))
#     pixel_grids = get_pixel_grids(height, width)
#     pixel_grids = expand_dims(pixel_grids, 0)
#     pixel_grids = tile(pixel_grids, [batch_size, 1])
#     pixel_grids = reshape(pixel_grids, (batch_size, 3, -1))
#     # return pixel_grids
#
#     # affine + divide tranform, output (B, 2, (W+1) x (H+1))
#     grids_affine = ops.matmul(affine_mat, pixel_grids)
#     grids_div = ops.matmul(div_mat, pixel_grids)
#     grids_zero_add = cast(equal(grids_div, 0.0), mstype.float32) * 1e-7  # handle div 0
#     grids_div = grids_div + grids_zero_add
#     grids_div = tile(grids_div, [1, 2, 1])
#     grids_inv_warped = div(grids_affine, grids_div)
#     x_warped, y_warped = unstack_axis_1(grids_inv_warped)
#     x_warped_flatten = reshape(x_warped, [-1])
#     y_warped_flatten = reshape(y_warped, [-1])
#
#     # interpolation
#     warped_image = interpolate(input_image, x_warped_flatten, y_warped_flatten)
#     warped_image = reshape(warped_image, input_image.shape)
#
#     # return input_image
#     return warped_image

