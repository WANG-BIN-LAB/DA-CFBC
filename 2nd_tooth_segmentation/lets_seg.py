import os
import sys
import re
import argparse
import torch
import nibabel as nib
import numpy as np         
from single_tooth_segmentation.networks.DAUNet_1st import daunet_tiny as DAUNet_1st
from single_tooth_segmentation.networks.DAUNet_2nd import daunet_tiny as DAUNet_2nd
from cnt_skl_dect import cnt_detection
from skimage.segmentation import find_boundaries
from single_tooth_segmentation.tooth_instance_seg import ins_tooth_seg

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str,  default='0, 1, 2, 3', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

num_classes = 2

with open('/home/data2/DA-CFBC/Adataset/dataset/img.list', 'r') as f:
    image_list = f.readlines()
image_list = [item.replace('\n','') for item in image_list]

with open('/home/data2/DA-CFBC/Adataset/dataset/label.list', 'r') as f:
    label_list = f.readlines()
label_list = [item.replace('\n','') for item in label_list]


def read_data(data_patch,label_patch):
    src_data_file = os.path.join(data_patch)
    label_file = os.path.join(label_patch)

    src_data_vol = nib.load(src_data_file)
    label_data_vol = nib.load(label_file)

    image = src_data_vol.get_fdata()
    label = label_data_vol.get_fdata()

    label_bin = label.copy()
    label_bin[label_bin > 1] = 1

    bg_area = np.where(label_bin < 1, label_bin, 0)
    teeth_area = label_bin - bg_area 
    edge = find_boundaries(teeth_area, mode='inner').astype(np.int16)

    multi_label = label.copy()

    w, h, d = image.shape
    spacing = src_data_vol.header['pixdim'][1:4]#0.25
    image = label_rescale(image, w*(spacing[0]/0.2), h*(spacing[0]/0.2), d*(spacing[0]/0.2), 'nearest')
    label = label_rescale(label, w*(spacing[0]/0.2), h*(spacing[0]/0.2), d*(spacing[0]/0.2), 'nearest')
    multi_label = label_rescale(multi_label, w*(spacing[0]/0.2), h*(spacing[0]/0.2), d*(spacing[0]/0.2), 'nearest')
    label_bin = label_rescale(label_bin, w*(spacing[0]/0.2), h*(spacing[0]/0.2), d*(spacing[0]/0.2), 'nearest')
    low_bound = np.percentile(image, 5)
    up_bound = np.percentile(image, 99.9)

    return image, label_bin, multi_label, edge, low_bound, up_bound, w, h, d


def load_model():
    
    net_cnt = DAUNet_1st().cuda()
    save_mode_path = os.path.join('')
    net_cnt.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net_cnt.eval()

    ins_net = DAUNet_2nd().cuda()
    save_mode_path = os.path.join('')
    print("init weight from {}".format(save_mode_path))
    ins_net.load_state_dict(torch.load(save_mode_path))
    ins_net.eval()

    return net_cnt, ins_net

def label_rescale(image_label, w_ori, h_ori, z_ori, flag):
    w_ori, h_ori, z_ori = int(w_ori), int(h_ori), int(z_ori)
    if flag == 'trilinear':
        teeth_ids = np.unique(image_label)
        image_label_ori = torch.zeros((w_ori, h_ori, z_ori)).cuda()
        image_label = torch.from_numpy(image_label).cuda()
        for label_id in range(len(teeth_ids)):
            image_label_bn = (image_label == teeth_ids[label_id]).float()
            image_label_bn = image_label_bn[None, None, :, :, :]
            image_label_bn = torch.nn.functional.interpolate(image_label_bn, size=(w_ori, h_ori, z_ori), mode='trilinear')
            image_label_bn = image_label_bn[0, 0, :, :, :]
            image_label_ori[image_label_bn > 0.5] = teeth_ids[label_id]
        image_label = image_label_ori.cpu().data.numpy()
    
    if flag == 'nearest':
        image_label = torch.from_numpy(image_label.astype(float))
        image_label = image_label[None, None, :, :, :]
        image_label = torch.nn.functional.interpolate(image_label, size=(w_ori, h_ori, z_ori), mode='nearest')
        image_label = image_label[0, 0, :, :, :].numpy()
    return image_label

def inference(image, label_bin, multi_label, edge, net_cnt, ins_net, low_bound, up_bound, w_o, h_o, d_o,data_id,image_list):
    w, h, d = image.shape
    print('---run the 1st stage network.')
    stride_xy = 64
    stride_z = 64
    patch_size_1st_stage = (128, 128, 128)
    centroids,cnt_test = cnt_detection(net_cnt, image[0:w:2, 0:h:2, 0:d:2], stride_xy, stride_z, patch_size_1st_stage,data_id,image_list)
    flag = True
    if centroids.size == 0 or cnt_test.size == 0:
        flag = False
        print("No centroids or cnt_test found after clustering, skipping...")
    else:
        cnt_test = label_rescale(cnt_test, w, h, d, 'nearest')
        cnt = nib.Nifti1Image(cnt_test, affine=np.eye(4))

    # 1st stage 
    print('---run the 2nd stage network.')
    patch_size = np.array([80, 80, 128])
    cnt_coords = centroids
    ins_skl_map = multi_label
    if flag:
        tooth_label = ins_tooth_seg(ins_net, image, label_bin, multi_label, cnt_coords, cnt_test, ins_skl_map, patch_size)
        whole_label = np.zeros((w, h, d))
        whole_label = tooth_label
        whole_label = label_rescale(whole_label, w_o, h_o, d_o, 'trilinear')
        return whole_label
    else:
        return None
        


if __name__ == '__main__':
    net_cnt, ins_net = load_model()
    for data_id in range(len(image_list)):
        print('**********process the data:', data_id)
        image, label_bin, multi_label, edge, low_bound, up_bound, w_o, h_o, d_o = read_data(image_list[data_id],label_list[data_id])
        tooth_label = inference(image, label_bin, multi_label, edge, net_cnt, ins_net, low_bound, up_bound, w_o, h_o, d_o,data_id,image_list)

        path_pos_0 = [sub_data_path.start() for sub_data_path in re.finditer('/', image_list[data_id])][-3]
        path_pos_1 = [sub_data_path.start() for sub_data_path in re.finditer('/', image_list[data_id])][-2]
        path_pos_2 = [sub_data_path.start() for sub_data_path in re.finditer('/', image_list[data_id])][-1]
        path_pos_3 = [sub_data_path.start() for sub_data_path in re.finditer('.nii.gz', image_list[data_id])][-1]

        save_path = '/home/data2/DA-CFBC/result_2025/'#2nd 33153_ms 3rd gc0125_caa
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if tooth_label is not None:
            nib.save(nib.Nifti1Image(tooth_label.astype(np.float32), np.eye(4)), save_path + 'pred_' + image_list[data_id][(path_pos_2+1):path_pos_3] + ".nii.gz")
            print(str(image_list[data_id][(path_pos_2+1):path_pos_3]))
        else:
            print(print(str(image_list[data_id][(path_pos_2+1):path_pos_3])),"该case没有检测到牙齿")
    print("!!")
