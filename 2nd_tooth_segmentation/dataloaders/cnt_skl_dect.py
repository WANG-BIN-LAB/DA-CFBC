import torch
import math
import numpy as np
import torch.nn.functional as F
from scipy import ndimage



def filter_connected_components(seg, min_size=500):

    # Label connected components
    from scipy import ndimage
    labeled_seg, num_features = ndimage.label(seg)

    # Create a mask to remove small connected components
    component_sizes = ndimage.sum(seg, labeled_seg, range(num_features + 1))
    mask = component_sizes >= min_size

    # Apply the mask to the labeled segmentation
    filtered_seg = np.where(labeled_seg == 0, 0, labeled_seg * mask[labeled_seg])
    filtered_seg[filtered_seg>1] = 1
    return filtered_seg

def cen_cluster(seg, off):
    centroids = np.array([])
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0
    voting_map = np.zeros(seg.shape)
    coord = np.array(np.nonzero((seg == 1)))
    coord = coord + off[:, seg == 1]
    coord = coord.astype(int)
    coord, coord_count = np.unique(coord, return_counts=True, axis=1)
    np.clip(coord[0], 0, voting_map.shape[0] - 1, out=coord[0])
    np.clip(coord[1], 0, voting_map.shape[1] - 1, out=coord[1])
    np.clip(coord[2], 0, voting_map.shape[2] - 1, out=coord[2])
    voting_map[coord[0], coord[1], coord[2]] = coord_count

    index_pts = (voting_map > 60)
    coord = np.array(np.nonzero((index_pts == 1)))
    num_pts = coord.shape[1]
    if num_pts < 1e1:
        return centroids, np.zeros(voting_map.shape)

    coord_dis_row = np.repeat(coord[:, np.newaxis, :], num_pts, axis=1)
    coord_dis_col = np.repeat(coord[:, :, np.newaxis], num_pts, axis=2)
    coord_dis = np.sqrt(np.sum((coord_dis_col - coord_dis_row) ** 2, axis=0))
    coord_score = voting_map[index_pts]
    coord_score_row = np.repeat(coord_score[np.newaxis, :], num_pts, axis=0)
    coord_score_col = np.repeat(coord_score[:, np.newaxis], num_pts, axis=1)
    coord_score = coord_score_col - coord_score_row

    coord_dis[coord_score > -0.5] = 1e10
    weight_dis = np.amin(coord_dis, axis=1)
    weight_score = voting_map[index_pts]
    centroids = coord[:, (weight_dis > 10) * (weight_score > 20)]

    if centroids.size == 0:
        return np.array([]), np.zeros(voting_map.shape)

    cnt_test = np.zeros(voting_map.shape)
    current_value = 1

    for i in range(centroids.shape[1]):
        centroid = (centroids[0, i], centroids[1, i], centroids[2, i])
        cnt_test[centroid[0], centroid[1], centroid[2]] = current_value
        current_value += 1
    cnt_test = ndimage.grey_dilation(cnt_test, size=(3, 3, 3))

    return centroids, cnt_test

def detect(net_cnt, image, stride_xy, stride_z, patch_size, data_id, image_list):
    w, h, d = image.shape

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
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad),(dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape


    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = np.zeros((2, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)
    cnt_off = np.zeros((3, ) + image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch_cnt = torch.from_numpy(test_patch).cuda()
                # network inference
                seg_cnt_patch, cnt_off_patch = net_cnt(test_patch_cnt)

                cnt_off_patch = cnt_off_patch.cpu().data.numpy()
                cnt_off[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] = cnt_off[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + cnt_off_patch[0,:,:,:,:]
                y = (F.softmax(seg_cnt_patch, dim=1).cpu().data.numpy()) 
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
   
    score_map = score_map/np.expand_dims(cnt,axis=0)
    cnt_off = cnt_off/np.expand_dims(cnt,axis=0)
    score_map = (score_map[1, :, :, :] > 0.9).astype(np.float32)
    label_map = score_map
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        cnt_off = cnt_off[:, wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    label_map = filter_connected_components(label_map, min_size=500)
    centroids,cnt_test = cen_cluster(label_map, cnt_off) 
    return centroids,cnt_test
    
    
def cnt_skl_detection(net_cnt, image, stride_xy, stride_z, patch_size,data_id,image_list):

    centroids,cnt_test = detect(net_cnt, image, stride_xy, stride_z, patch_size,data_id,image_list)
    return centroids,cnt_test
    
