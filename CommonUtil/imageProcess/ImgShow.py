import matplotlib.pyplot as plt

'''
显示多张图片
'''


def showLineImg(imgList,figsize=(10,6),title=[]):
    length = len(imgList)
    fig,ax = plt.subplots(1,length,figsize = figsize)
    for i in range(length):
        ax[i].imshow(imgList[i])
        if len(title) == len(imgList):
            ax[i].set_title(title[i],fontsize=12,color='r')




