import numpy as np
import scipy.io as scio
import cv2
'''
this file have some code about pre-processing some image
'''

# 读取mat文件
def readMatFile(matFilePath):
    matFile = scio.loadmat(matFilePath)
    return matFile
    
# 使用opencv读取文件并且转化成RGB图像
def cv2Bgr2Rgb(imgPath):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


    
# -------------------------------------------------------下面是brp-net里面用到的一些函数---------------------------------------------------------
# 图像的变形        
def transformImg(img):
    norm_mean=[0.485, 0.456, 0.406]
    norm_std=[0.229, 0.224, 0.225]

    _, _, mod = img.shape
    img = Image.fromarray(img.astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(seed())
    img = ImageEnhance.Brightness(img).enhance(seed())
    img = ImageEnhance.Contrast(img).enhance(seed())
    img = ImageEnhance.Sharpness(img).enhance(seed())
    img = np.asarray(img).astype(np.float32)
    img = img.transpose([2, 0, 1])
    for imod in list(range(mod)):
        img[imod] = (img[imod]/255.0 - norm_mean[imod])/norm_std[imod]
    img += np.random.normal(0, np.random.rand(), img.shape)*0.01
    return img

def test_extract_patches(img, patch_sz, stride):
    b, c,  h, w = img.shape
    padding_h = (patch_sz - stride)//2
    padding_w = (patch_sz - stride)//2
    padding_h_img = int(np.ceil(h/stride)*stride - h) // 2
    padding_w_img = int(np.ceil(w/stride)*stride - w) // 2
    pad_img = F.pad(img, (padding_w_img + padding_w, padding_w_img + padding_w, padding_h_img + padding_h, padding_h_img + padding_h), mode='reflect')
    return pad_img

# 一些提取图片的操作
# 里面包含了一些重叠采样的方法
# 0 - 256 - 0 256
# 0 - 256 - 128 384
# 0 - 256 - 256 512
# 0 - 256 - 384 640
# 0 - 256 - 512 768
# 0 - 256 - 640 896
# 0 - 256 - 768 1024
# 0 - 256 - 896 1152
def extract_patches(img, patch_sz, stride):
    b, c,  h, w = img.shape
    padding_h = (patch_sz - stride)//2
    padding_w = (patch_sz - stride)//2
    padding_h_img = int(np.ceil(h/stride)*stride - h) // 2
    padding_w_img = int(np.ceil(w/stride)*stride - w) // 2
    # reflect将右边的值镜像过来
    pad_img = F.pad(img, (padding_w_img + padding_w, padding_w_img + padding_w, padding_h_img + padding_h, padding_h_img + padding_h), mode='reflect')
    # [4, 3, 1152, 1152]
    _, _, h_pad, w_pad = pad_img.shape
    patches = []
    ibs = []
    shs = []
    sws = []
    imgCount = 0
    for ib in list(range(b)):
        for sh in list(range(padding_h, padding_h+h+padding_h_img*2, stride)):
            for sw in list(range(padding_w, padding_w+w+padding_w_img*2, stride)):
                tmp_p = pad_img[ib, :, (sh-padding_h):(sh+padding_h+stride), (sw-padding_w):(sw+padding_w+stride)].unsqueeze(dim=0)
                patches.append(tmp_p)
                ibs.append(ib)
                shs.append(sh)
                sws.append(sw)
    patches = torch.cat(tuple(patches), dim=0)
    return patches, ibs, shs, sws

# 从patch中重构图片
def reconstruct_from_patches_weightedall(patches, ibs, shs, sws, patch_sz, stride, b, c, h, w, patches_weight_map):
    padding_h = (patch_sz - stride)//2
    padding_w = (patch_sz - stride)//2
    padding_h_img = int(np.ceil(h/stride)*stride - h) // 2
    padding_w_img = int(np.ceil(w/stride)*stride - w) // 2
    img_rc = torch.zeros(b, c, h+2*padding_h_img+2*padding_h, w+2*padding_w_img+2*padding_w)
    ncount = torch.zeros(b, c, h+2*padding_h_img+2*padding_h, w+2*padding_w_img+2*padding_w)
    #if len(patches_weight_map.shape)==3:
    #    patches_weight_map = patches_weight_map.unsqueeze(dim=0)
    ipatches = 0
    for ipatches in list(range(patches.shape[0])):
        ib = ibs[ipatches]
        sh = shs[ipatches]
        sw = sws[ipatches]
        img_rc[ib, :, (sh-padding_h):(sh+padding_h+stride), (sw-padding_w):(sw+padding_w+stride)] += patches[ipatches] * patches_weight_map
        ncount[ib, :, (sh-padding_h):(sh+padding_h+stride), (sw-padding_w):(sw+padding_w+stride)] += patches_weight_map
    img_rc_norm =  img_rc / ncount
    img_rc_norm = img_rc_norm[:, :, (padding_h_img+padding_h):(padding_h_img+padding_h+h), (padding_w_img+padding_w):(padding_w_img+padding_w+w)]
    return img_rc_norm





