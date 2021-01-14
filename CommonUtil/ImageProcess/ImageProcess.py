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






