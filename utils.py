import sys
import os
from os.path import join
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import torch.nn as nn
import torch

from skimage.util import view_as_windows,pad
import numpy as np
from detectron2.structures import BoxMode
from glob import glob


class UVmapGenerator():
    def __init__(self, disp_dir):
        self.disp_dir = disp_dir
        # self.disp_ids = os.listdir(self.disp_dir)
        # self.num_disp_maps = len(self.disp_ids)
        self.name = disp_dir.split("/")[-1]

    def onehot_initialization(self, mat):
        '''
        to get a 3d matrix of the world view
        '''
        ncols = mat.max() + 1
        out = np.zeros(mat.shape + (ncols,), dtype=int)
        out[self.all_idx(mat, axis=2)] = 1
        return out

    def all_idx(self, idx, axis):
        '''
        to get the necessary indices
        '''
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def get_uv_disparity(self, depth_map):
        '''
        uses the depth_map to find u and v disparity
        '''
        # using disparity as a measure of depth, generating one-hot world-view
        world_view = self.onehot_initialization(depth_map)
        # project the world views to get disparity maps
        u_disparity = np.sum(world_view, axis=0).T
        v_disparity = np.sum(world_view, axis=1)
        return v_disparity, u_disparity

    def draw_hoglines(self, src, target='u'):
        dst = np.zeros(tuple(list(src.shape) + [3]))
        src = (src * 255).astype(np.uint8)
        edges = cv2.Canny(src, 70, 150, apertureSize=3)
        if target == 'u':
            linesP = cv2.HoughLinesP(edges, 1, np.pi / 2, 100, None, 10, 30)
        elif target == 'v':
            linesP = cv2.HoughLinesP(edges, 1, np.pi / 2, 100, None, 10, 30)

        if linesP is not None:
            linesP = np.squeeze(np.array(linesP))
            for i in range(linesP.shape[0]):
                l = linesP[i]
                try:
                    cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (1, 1, 1), 1, cv2.LINE_AA)
                except:
                    pass

        return dst, linesP

    def visualize_uvmaps(self):
        # indices = np.random.randint(self.num_disp_maps, size=num).tolist()

        fig = plt.figure(figsize=(30, 1 * 3 * 5))

        disp_map = cv2.imread(self.disp_dir)
        disp_map = cv2.cvtColor(disp_map, cv2.COLOR_BGR2GRAY)
        v_disp, u_disp = self.get_uv_disparity(disp_map)

        ax1 = fig.add_subplot(3, 1, 0 * 3 + 1)
        ax1.imshow(disp_map)
        ax1.set_title(f'{self.name}')
        ax1.axis('off')

        v_ret, _ = self.draw_hoglines(v_disp, target='v')
        ax2 = fig.add_subplot(1 * 3, 1, 0 * 3 + 2)
        ax2.imshow(v_ret, cmap='gray')
        ax2.set_title(f'V-Disp for {self.name}')
        ax2.axis('off')

        u_ret, _ = self.draw_hoglines(u_disp, target='u')
        ax3 = fig.add_subplot(1 * 3, 1, 0 * 3 + 3)
        ax3.imshow(u_ret, cmap='gray')
        ax3.set_title(f'U-Disp for {self.name}')
        ax3.axis('off')


class ObjectLocalizer(UVmapGenerator):
    def __init__(self, orig_dir, disp_dir):
        super().__init__(disp_dir)

        self.orig_dir = orig_dir
        # self.orig_ids = self.disp_ids
        # self.num_orig_maps = len(self.orig_ids)

    def compute_intersection(self, bb1, bb2):
        '''
        Args:
            `bb1`, `bb2`: contain bounding boxes's info in the following format:
                            [xmin, ymin, xmax, ymax]
        '''

        assert bb1[0] < bb1[2]
        assert bb1[1] < bb1[3]
        assert bb2[0] < bb2[2]
        assert bb2[1] < bb2[3]
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        return np.array([x_left, y_top, x_right, y_bottom])

    def compute_iou(self, bb1, bb2):
        '''
        Args:
            `bb1`, `bb2`: contain bounding boxes's info in the following format:
                            [xmin, ymin, xmax, ymax, (class_id)]
        '''
        assert bb1[0] < bb1[2]
        assert bb1[1] < bb1[3]
        assert bb2[0] < bb2[2]
        assert bb2[1] < bb2[3]
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0

        return iou

    def get_obstacle_boxes(self, disp_map, epsilon=3):
        v_disp, u_disp = self.get_uv_disparity(disp_map)

        v_ret, v_lines = self.draw_hoglines(v_disp, target='v')
        u_ret, u_lines = self.draw_hoglines(u_disp, target='u')

        obstacles = np.zeros((1, 4))

        for i in range(u_lines.shape[0]):
            l = u_lines[i, :]
            disp_val = l[1]

            # Extend u_line into strip with a format of [xmin, ymin, xmax, ymax]
            xmin = min(l[0], l[2])
            xmax = max(l[0], l[2])
            u_box = np.array([xmin, 0, xmax, disp_map.shape[0]])

            # Get corresponding value in V-Disparity
            if len(v_lines.shape) == 1:
                v_lines = v_lines.reshape((1, 4))
            mask = np.logical_and(
                v_lines[:, 0] >= disp_val - epsilon,
                v_lines[:, 0] <= disp_val + epsilon
            )
            corr_lines = v_lines[mask, :]
            n_lines = corr_lines.shape[0]

            # Bounding boxes for corresponding V-Disparity lines
            x1 = np.array([0] * n_lines).reshape((n_lines, 1))
            y1 = np.min(corr_lines[:, [1, 3]], axis=1, keepdims=True)
            x2 = np.array([disp_map.shape[1]] * n_lines).reshape((n_lines, 1))
            y2 = np.max(corr_lines[:, [1, 3]], axis=1, keepdims=True)
            corr_boxes = np.hstack([x1, y1, x2, y2])
            if corr_boxes.shape[0] > 0:
                intersects = np.apply_along_axis(self.compute_intersection, 1, corr_boxes, u_box)
                obstacles = np.vstack([obstacles, intersects])

        return obstacles

    def draw_annotations(self, annos, ax):
        cmap = plt.cm.get_cmap('hsv', annos.shape[0])

        for i, (xmin, ymin, xmax, ymax) in enumerate(annos):
            w = xmax - xmin
            h = ymax - ymin
            rect = mpatches.Rectangle(
                (xmin, ymin), w, h,
                fill=False,
                edgecolor=cmap(i),
                linewidth=1.2
            )
            ax.add_patch(rect)

    def segment(self, src, annos):
        key = random.sample(list(range(256)), 3)

        # Fill with computed boxes
        for i, (xmin, ymin, xmax, ymax) in enumerate(annos):
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            src[ymin:ymax, xmin:xmax, 0] = key[0]
            src[ymin:ymax, xmin:xmax, 1] = key[1]
            src[ymin:ymax, xmin:xmax, 2] = key[2]

        mask = np.logical_and(
            src[:, :, 0] == key[0],
            src[:, :, 1] == key[1],
            src[:, :, 2] == key[2]
        )
        segmented = np.where(mask, 255, 0)

        return segmented / 255

    def get_mask(self):
        disp_map = cv2.imread(self.disp_dir)
        disp_map = cv2.cvtColor(disp_map, cv2.COLOR_BGR2GRAY)
        src = cv2.imread(self.orig_dir)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        annos = self.get_obstacle_boxes(disp_map)
        segmented = self.segment(src, annos)
        return segmented

    def visualize(self):
        # indices = np.random.randint(self.num_disp_maps, size=num).tolist()

        fig = plt.figure(figsize=(30, 3 * 6))

        ax1 = fig.add_subplot(1 * 3, 1, 1)
        disp_map = cv2.imread(self.disp_dir)
        disp_map = cv2.cvtColor(disp_map, cv2.COLOR_BGR2GRAY)
        ax1.imshow(disp_map)
        ax1.set_title(f'{self.name}')
        ax1.axis('off')

        ax2 = fig.add_subplot(1 * 3, 1, 2)
        src = cv2.imread(self.orig_dir)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        annos = self.get_obstacle_boxes(disp_map)
        ax2.imshow(src)
        self.draw_annotations(annos, ax2)
        ax2.axis('off')

        ax3 = fig.add_subplot(1 * 3, 1, 3)
        segmented = self.segment(src, annos)
        ax3.imshow(segmented)
        ax3.axis('off')




def bin2int(binary_list):
    """
    Convert binary array list to integer (decimal)
    :param binary_list: list of binary integer i.e. [1,0,1]
    :return: decimal equivalent value of binary numbers i.e. 5 for [1,0,1]
    """
    return binary_list.dot(1 << np.arange(binary_list.shape[-1] - 1, -1, -1))


def derivate_image(im, angle):
    '''
    Compute derivative of input image
    ###################################################################
    Implemented by: Adrian Ungureanu - June 2019
    https://github.com/AdrianUng/palmprint-feature-extraction-techniques
    Modified by: Omar Hassan - August 2020
    ###################################################################
    :param im: input image. should be grayscale!
    :param angle: 0 or 90 degrees
    :return: computed derivative along that direction.
    includes padding...
    '''
    h, w = np.shape(im)
    pad_im = np.zeros_like(im)
    if angle == 'horizontal':  # horizontal derivative
        pad_im[:,:w-1] = im[:,1:]
        pad_im[:,-1] = im[:,-1]
        deriv_im = pad_im - im  # [1:, :w]
    elif angle == 'vertical':
        pad_im[1:,:] = im[:h-1,:]
        pad_im[0,:] = im[0,:]
        deriv_im = pad_im - im  # [1:, :w]
    return deriv_im

###################################################################
def ltrp_first_order(im_d_x, im_d_y):
    """
    Extract LTrP1 code (4 orientations) by using input dx and dy matrices.
    ###################################################################
    Implemented by: Adrian Ungureanu - June 2019
    https://github.com/AdrianUng/palmprint-feature-extraction-techniques
    ###################################################################
    :param im_d_x: derivative of image according to x axis (horizontal)
    :param im_d_y: derivative of image according to y axis (vertical)
    :return: encoded LTrP1 code. Possible values ={1,2,3,4}
    """
    encoded_image = np.zeros(np.shape(im_d_y))  # define empty matrix, of the same shape as the image...

    # # apply conditions for each orientation...
    encoded_image[np.bitwise_and(im_d_x >= 0, im_d_y >= 0)] = 1
    encoded_image[np.bitwise_and(im_d_x < 0, im_d_y >= 0)] = 2
    encoded_image[np.bitwise_and(im_d_x < 0, im_d_y < 0)] = 3
    encoded_image[np.bitwise_and(im_d_x >= 0, im_d_y < 0)] = 4

    return encoded_image



def ltrp_second_order_fast(ltrp1):
    """
    Faster implementation to compute second order ltrp code. The code
    only computes tetra patterns and does not compute magnitude pattern
    however the code may be extended to compute magnitude as well following
    similar structure.
    :param ltrp1: first order ltrp code
    :return: second order ltrp code
    #######################################################
    Implemented by Omar Hassan - August 2020
    #######################################################
    """
    #get img dimension, currently only supports square dim
    im_side0 = ltrp1.shape[0]
    im_side1 = ltrp1.shape[1]
    #pad input with zeros
    ltrp1 = np.pad(ltrp1, (1, 1), 'constant', constant_values=0)
    #convert to type int to save memory used in next step
    ltrp1 = ltrp1.astype(np.int)
    #generate patches that will be used for performing ltrp steps
    patches = view_as_windows(ltrp1,window_shape=(3,3),step=1)
    #reshape patches to 3x3 shape, same as kernel size
    patches = patches.reshape(-1,3,3)

    #center pixels
    g_c = patches.copy()[:,1,1]
    g_c = g_c.astype(np.uint8)
    g_c = g_c.reshape(-1,1)

    #set center pixels = -1 to later filter and remove from neighbor array
    patches[:,1,1] = -1

    #reshape to vector
    neighbor = patches.reshape(patches.shape[0],-1)

    #retain original number of patches
    patch_num = neighbor.shape[0]

    #reshape to perform boolwise element selection
    neighbor = neighbor.reshape(-1,)
    #filter out array values if they equal to -1
    nmask = neighbor==-1
    neighbor = neighbor[~nmask]
    #reshape back to original number of patches
    neighbor = neighbor.reshape(patch_num,-1)
    #convert to uint8
    neighbor = neighbor.astype(np.uint8)

    #if central pixel equals neighbor in every patches, set them to 0
    mask = neighbor != g_c
    neighbor = np.multiply(neighbor,mask)

    #for ltrp, we generate feature maps for 4(tetra) directions,
    #for each direction, we exclude its value from the patches
    #similarly, we also exclude center pixel of patch
    directions = np.array([[2, 3, 4],
                            [1, 3, 4],
                            [1, 2, 4],
                            [1, 2, 3]])

    #our original patch are indexed as [0,1,2,3,4,5,6,7] (g_c and current_direction pixels are removed from patch)
    #where 0 is top-left, and 7 is bottom right
    #in algorithm, indexed are done as [3,2,1,4,0,5,6,7]
    #where 0 is mid-right, and 7 is bottom right
    #hence we need to swap our indexes

    # swap_idx = np.array([3,2,1,4,0,5,6,7]) # Original Author uses these indexes

    swap_idx = np.array([6,5,3,0,1,2,4,7])  # Referenced Implementation uses these indexes

    neighbor = neighbor[:,swap_idx]

    g_c1234 = []

    for i in range(4):
        #get an array where only central pixel == 1 are there, rest are zero
        #we do this for doing computations related to centeral_pixel==1
        gc_i = g_c.copy()
        #since loop starts from 0, mask index is i+1 here.
        #(i+1) actually means current direction
        #for each direction, we will compute feature maps
        gc_i[gc_i!=(i+1)] = 0
        #reshape to 1d-array
        gc_i = gc_i.reshape(-1,)

        temp = neighbor.copy()

        masks = [temp[gc_i.astype(np.bool)] == x for x in directions[i]]

        pattern = [bin2int(x) for x in masks]

        pattern = np.array(pattern)

        g_c_img = np.zeros((3,im_side0*im_side1),dtype=np.uint8)

        g_c_img[:,gc_i.astype(np.bool)] = pattern

        g_c1234.append(g_c_img)

    g_c1234 = np.array(g_c1234)

    g_c1234 = g_c1234.reshape(-1,im_side0,im_side1)

    return g_c1234

def get_ltrp(image):
    """
    Computes ltrp as a whole (both first and second order).
    :param image: input grayscale image
    :param original: whether to use Adrian's implementation or mine
    :return: 12 ltrp feature maps
    """

    image = np.array(image, dtype=np.float32)  # /255.

    deriv_h = derivate_image(image, 'horizontal')
    deriv_v = derivate_image(image, 'vertical')


    ltrp1 = ltrp_first_order(im_d_x=deriv_h, im_d_y=deriv_v)
    ltrp2 = ltrp_second_order_fast(ltrp1)

    return ltrp2

def concat_fm(fm):
    """
    Concatenates Directional feature maps as shown in original paper.
    This function is used for visualization purposes only.
    :param fm: 12 ltrp feature maps
    :return: list of 4 concatenated feature maps
    """

    d1 = fm[0]+fm[1]+fm[2]
    d2 = fm[3]+fm[4]+fm[5]
    d3 = fm[6]+fm[7]+fm[8]
    d4 = fm[9]+fm[10]+fm[11]

    return [d1,d2,d3,d4]
def LTrP_complete(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ffm = get_ltrp(gray)
  fm_list = concat_fm(ffm)
  final_pro = fm_list[0] + fm_list[1]+fm_list[2]+fm_list[3]
  final_pro=cv2.merge([final_pro,final_pro,final_pro])
  return final_pro



CLASSES = {'Car' : 0, 'Van' : 1, 'Truck' : 2,
                     'Pedestrian' : 3, 'Person_sitting' : 4, 'Cyclist' :5, 'Tram':6,
                     'Misc':7, 'DontCare':8}
def get_kitti_dicts(img_dir, label_dir, disp_dir=None, using_mask=False):

    labels_list = glob(label_dir + "/*.txt")

    dataset_dicts = []
    for i, label_file in enumerate(labels_list):
        f = open(label_file, "r")
        lines = f.readlines()
        record = {}
        file_name = label_file.split("/")[-1]
        file_name = file_name[:-4]
        file_dir = img_dir + "/" + file_name + ".png"
        if disp_dir is not None:
            disp_name = disp_dir + "/"+file_name+".png"
        img = cv2.imread(file_dir)
        if img is None:
            continue
        H, W = img.shape[:2]

        # modified for adaptive ratio LP
        LP_image = LTrP_complete(img)
        record["LP_image"] = LP_image
        if using_mask:
            object_localizer = ObjectLocalizer(file_dir,disp_dir)
            mask = object_localizer.get_mask()
        record['mask'] = mask

        record["file_name"] = file_dir
        record["image_id"] = i
        record["height"] = H
        record["width"] = W
        print(record)
        # cv2_imshow(img)
        objs = []
        for line in lines:
            annos = line[:-1].split(" ")
            class_id = CLASSES[annos[0]]
            bb = [float(i) for i in annos[4:8]]

            obj = {
                "bbox": bb,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_id,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def spatial_pyramid_pool(images, size, mask = None):
  H, W = images.shape[-2:]
  padded_mask = None
  h_wid = int(math.ceil(H/size))
  w_wid = int(math.ceil(W/size))
  h_pad_left = (h_wid*size - H)//2
  h_pad_right = (h_wid*size - H) - h_pad_left
  w_pad_left = (w_wid*size-W) //2
  w_pad_right = ((w_wid*size-W)) - w_pad_left
  pad = (w_pad_right, w_pad_left, h_pad_right, h_pad_left,0,0)
  padded_image = nn.functional.pad(images, pad, "constant", 0)
  if mask is not None:
    padded_mask = nn.functional.pad(mask, pad, "constant", 0)
  # print(padded_image.shape)
  return padded_image, padded_mask, h_wid, w_wid

def build_mask(segment, s=15,kernel_size = (24,77), Thresh = 0.75):
    patches = torch.nn.functional.unfold(segment.float(), kernel_size=kernel_size, stride=kernel_size)
    patches = patches.reshape((1, -1, s, s))
    mean = patches.mean(dim=1)
    mask = (mean>Thresh)*1
    return mask