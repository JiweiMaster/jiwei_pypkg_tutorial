import numpy as np
from skimage.morphology import label
from scipy import ndimage
import scipy.io as scio
import cv2

def get_rect_of_mask(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax
    
def get_size_of_mask(img):
    if np.max(img) == 0:
        return 0
    rmin, rmax, cmin, cmax = get_rect_of_mask(img)
    return max([rmax - rmin, cmax - cmin])
    
def remove_overlaps(instances, scores):
    if len(instances) == 0:
        return [], []
    lab_img = np.zeros(instances[0].shape, dtype=np.int32)
    for i, instance in enumerate(instances):
        lab_img = np.maximum(lab_img, instance * (i + 1))
    instances = []
    new_scores = []
    for i in range(1, lab_img.max() + 1):
        instance = (lab_img == i).astype(np.bool)
        if np.max(instance) == 0:
            continue
        instances.append(instance)
        new_scores.append(scores[i - 1])
    return instances, new_scores
    
def post_proc(output, cutoff=0.5, cutoff_instance_max=0.3, cutoff_instance_avg=0.2, post_dilation_iter=2, post_fill_holes=True):
    """
    Split 1-channel merged output for instance segmentation
    :param cutoff:
    :param output: (h, w, 1) segmentation image
    :return: list of (h, w, 1). instance-aware segmentations.
    """
    # The post processing function 'post_proc' is borrowed from the author of CIA-Net.
    
    cutoffed = output > cutoff
    # 标记整数数组的连接区域,获取连通区域
    lab_img = label(cutoffed, connectivity=1)
    instances = []
    # pdb.set_trace()
    for i in range(1, lab_img.max() + 1):
        #将每个连通分类对应的单独的图片提取出来
        instances.append((lab_img == i).astype(np.bool))

    filtered_instances = []
    scores = []
    for instance in instances:
        # TODO : max or avg?
        # 这边所谓的score得分就是代表的有效连通分量的面积，太小的话就直接过滤掉代表无效的区域不要了
        instance_score_max = np.max(instance * output)    # score max
        if instance_score_max < cutoff_instance_max:
            continue
        instance_score_avg = np.sum(instance * output) / np.sum(instance)   # score avg
        if instance_score_avg < cutoff_instance_avg:
            continue
        filtered_instances.append(instance)
        scores.append(instance_score_avg) # 使用的是score_avg，用来最后算instance大小的
    instances = filtered_instances # 过滤掉太小的

    # dilation对二值图像进行膨胀操作，消除原来的空洞
    instances_tmp = []
    if post_dilation_iter > 0:
        for instance in filtered_instances:
            instance = ndimage.morphology.binary_dilation(instance, iterations=post_dilation_iter)
            instances_tmp.append(instance)
        instances = instances_tmp

    # sorted by size
    sorted_idx = [i[0] for i in sorted(enumerate(instances), key=lambda x: get_size_of_mask(x[1]))]
    instances = [instances[x] for x in sorted_idx]
    scores = [scores[x] for x in sorted_idx]

    # make sure there are no overlaps
    # todo: this dataset gt has overlap, so do not use this func
    # 移除重叠的部分
    instances, scores = remove_overlaps(instances, scores)

    # fill holes, 填补一些因为腐蚀而产生的洞
    if post_fill_holes:
        instances = [ndimage.morphology.binary_fill_holes(i) for i in instances]
    
    # instances = [np.expand_dims(i, axis=2) for i in instances]
    # scores = np.array(scores)
    # scores = np.expand_dims(scores, axis=1)
    if len(instances) > 0:
        lab_img = np.zeros(instances[0].shape, dtype=np.int32)
        for i, instance in enumerate(instances):
            lab_img = np.maximum(lab_img, instance * (i + 1))
        return lab_img
    else:
        return None

# 获取label图里面的instance
def getInstancePosition(gt_label,margin=12):
# image position => (y:y+h,x:x+w)
    max = np.max(gt_label)
    instanceList = []
    for idx in range(1,max):
        gt_copy = gt_label.copy()
        gt_copy[gt_copy != idx] = 0
        gt_copy = gt_copy.astype('uint8')
        x,y,w,h = cv2.boundingRect(gt_copy)
        # instanceList.append([y-margin,y+h+margin,x-margin,x+w+margin])
        instanceList.append([x,y,w,h])
    return np.array(instanceList)


# 未使用opencv的方法获取instance的坐标
def getInstanceArray(gt_label):
    # 所有的instance的大小
    predset  = np.unique(gt_label[gt_label>0])
    instanceList = []
    for ic in predset:
        instance = gt_label.copy()
        instance[ instance != ic ] = 0
        instance[ instance == ic ] = 1
        # 获取不为0的位置
        icx, icy = np.nonzero(instance)
        maxx = icx.max()
        maxy = icy.max()
        minx = icx.min()
        miny = icy.min()
        mx = np.round((maxx+minx)/2)
        my = np.round((maxy+miny)/2)
        halfsz = (np.max([(maxx-minx)/2, (maxy-miny)/2, 8])+12).astype(np.int16)
        # 图像实际的起始位置
        sx = np.round(mx - halfsz).astype(np.int16)
        sy = np.round(my - halfsz).astype(np.int16)
        ex = np.round(mx + halfsz + 1).astype(np.int16)
        ey = np.round(my + halfsz + 1).astype(np.int16)
        # 添加instance的数据的，获取的是方形的数据
        instanceList.append([sx,sy,ex,ey])
    return instanceList


# 根据二值图来获取instance的起止坐标
def getInstancePosi(binaryImg):
    # 获取不为0的位置
    icx, icy = np.nonzero(binaryImg)
    maxx = icx.max()
    maxy = icy.max()
    minx = icx.min()
    miny = icy.min()
    mx = np.round((maxx+minx)/2)
    my = np.round((maxy+miny)/2)
    halfsz = (np.max([(maxx - minx)/2, (maxy - miny)/2, 8])+12).astype(np.int16)
    # 图像实际的起始位置
    sx = np.round(mx - halfsz).astype(np.int16)
    sy = np.round(my - halfsz).astype(np.int16)
    ex = np.round(mx + halfsz + 1).astype(np.int16)
    ey = np.round(my + halfsz + 1).astype(np.int16)
    # 添加instance的数据的，获取的是方形的数据
    return sx,sy,ex,ey





