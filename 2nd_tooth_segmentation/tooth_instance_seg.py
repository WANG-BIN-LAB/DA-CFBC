import torch
import numpy as np
import cc3d
import torch.nn.functional as F
from skimage import morphology
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize_3d
from find_adj_cnt import find_nearest_points
from process_label import process_labels


def tooth_seg(net_seg, image, label_bin, multi_label, cnt_coords,centroids, ins_skl_map, patch_size):
    w, h, d = image.shape
    label = label_bin
    multi_skeleton = ins_skl_map
    
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
        label = np.pad(label, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
        multi_skeleton = np.pad(multi_skeleton, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
        centroids = np.pad(centroids, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape
    
    crop_size = np.array([80,80,128])
    image_list, label_list, centroids_list, crop_coord_min_list, center_list = [], [], [], [], [], []
    centroids_rev = cnt_coords.transpose(1,0)/(0.2/0.4)
    teeth_ids = np.unique(centroids)
    for i in range(len(teeth_ids)):
        tooth_id = teeth_ids[i]
        if tooth_id == 0:
            continue
        coord_cnt = np.nonzero((centroids == tooth_id))
        meanx1 = int(np.mean(coord_cnt[0]))
        meany1 = int(np.mean(coord_cnt[1]))
        meanz1 = int(np.mean(coord_cnt[2]))
        mean_coord_cnt = (meanx1, meany1, meanz1)

        mean_coord = mean_coord_cnt
        center_list.append(mean_coord)
        
        # generate the crop coords
        crop_coord_min = mean_coord - crop_size/2
        np.clip(crop_coord_min, (0, 0, 0), image.shape - crop_size, out = crop_coord_min)
        crop_coord_min = crop_coord_min.astype(int)


        for i in range(3):
            if((crop_coord_min[i]+crop_size[i])>image.shape[i]):
                crop_coord_min[i] = image.shape[i]-crop_size[i]

        crop_centroid = (centroids[crop_coord_min[0]:(crop_coord_min[0]+crop_size[0]), crop_coord_min[1]:(crop_coord_min[1]+crop_size[1]), crop_coord_min[2]:(crop_coord_min[2]+crop_size[2])] == tooth_id).astype(np.uint8)
        crop_centroid = skeletonize_3d(crop_centroid)
        crop_centroid = ndimage.grey_dilation(crop_centroid, size= (3, 3, 3))
        crop_centroid = morphology.remove_small_objects(crop_centroid.astype(bool), min_size=50, connectivity=1)
        crop_centroid = gaussian_filter(crop_centroid.astype(float), sigma=2)
        centroids_list.append(crop_centroid)

        image_list.append(image[crop_coord_min[0]:(crop_coord_min[0]+crop_size[0]), crop_coord_min[1]:(crop_coord_min[1]+crop_size[1]), crop_coord_min[2]:(crop_coord_min[2]+crop_size[2])])
        label_list.append(label[crop_coord_min[0]:(crop_coord_min[0]+crop_size[0]), crop_coord_min[1]:(crop_coord_min[1]+crop_size[1]), crop_coord_min[2]:(crop_coord_min[2]+crop_size[2])])
        crop_coord_min_list.append(crop_coord_min)

    patches_coord_min = np.asarray(crop_coord_min_list)
    patches_center = np.asarray(center_list)
    image_patches = np.asarray(image_list)
    label_patches = np.asarray(label_list)
    centroids_patches = np.asarray(centroids_list)
    image_patches = torch.from_numpy(image_patches[:, None, :, :, :]).float().cuda()
    label_patches = torch.from_numpy(label_patches[:, None, :, :, :]).float().cuda()
    centroids_patches = torch.from_numpy(centroids_patches[:, None, :, :, :]).float().cuda()
    with torch.no_grad():
        seg_patches_1,bd_patches_1 = net_seg(image_patches[:10, :, :, :, :])
        seg_patches_2,bd_patches_2 = net_seg(image_patches[10:20, :, :, :, :])
        seg_patches_3,bd_patches_3 = net_seg(image_patches[20:, :, :, :, :])

        seg_patches = torch.cat((seg_patches_1, seg_patches_2), 0)
        seg_patches = torch.cat((seg_patches, seg_patches_3), 0)
        bd_patches = torch.cat((bd_patches_1, bd_patches_2), 0)
        bd_patches = torch.cat((bd_patches, bd_patches_3), 0)

    seg_patches = F.softmax(seg_patches, dim=1)
    bd_patches = F.softmax(bd_patches, dim=1)
    seg_patches = torch.argmax(seg_patches, dim = 1)
    bd_patches = torch.argmax(bd_patches, dim = 1)
    seg_patches = seg_patches.cpu().data.numpy()
    bd_patches = bd_patches.cpu().data.numpy()

    seg_pathes_gt_test = seg_patches
    w2, h2, d2 = image.shape
    count = 0
    image_label = np.zeros((w2, h2, d2), dtype=int)
    image_vote_flag = np.zeros((w2, h2, d2), dtype=int)
    for crop_i in range(patches_coord_min.shape[0]):
        
        center_point = patches_center[crop_i]
        nearest_points = find_nearest_points(patches_center, center_point)

        labels = seg_pathes_gt_test[crop_i, :, :, :]
        labels_post = process_labels(crop_i,labels,bd_patches[crop_i,:,:,:],crop_size,nearest_points,patches_coord_min)
        labels = cc3d.connected_components(labels_post, connectivity=6)
        num = np.max(labels) + 1

        if num > 1:
            max_num = -1e10
            for lab_id in range(1, num+1):
                if np.sum(labels == lab_id) > max_num:
                    max_num = np.sum(labels == lab_id)
                    true_id = lab_id
            seg_pathes_gt_test[crop_i, :, :, :] = (labels == true_id)
        
        coord = np.array(np.nonzero((seg_pathes_gt_test[crop_i, :, :, :] == 1)))
        coord[0] = coord[0] + patches_coord_min[crop_i, 0]
        coord[2] = coord[2] + patches_coord_min[crop_i, 2]             
        image_vote_flag[coord[0], coord[1], coord[2]] = 1
        count = count + 1
        image_label[coord[0], coord[1], coord[2]] = count
        image_vote_flag[coord[0], coord[1], coord[2]] = 0
        
    print("count:",count)

    if add_pad:
        image_label = image_label[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return image_label

def ins_tooth_seg(ins_net, image, label_bin, multi_label, cnt_coords,centroids, ins_skl_map, patch_size):

    label_map = tooth_seg(ins_net, image, label_bin, multi_label, cnt_coords, centroids, ins_skl_map, patch_size)
    return label_map
