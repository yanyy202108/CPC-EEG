from PIL import Image
#import os
#import os.path
import glob
import cv2 as cv
import re
#path = "D:\\contrastive-predictive-coding-master\\images\\yuan"
#all_images = os.listdir(path)
# print(all_images)

#for image in all_images:
#    image_path = os.path.join(path, image)
#    img = Image.open(image_path) # 打开图片
#    img = img.convert("RGB")  # 4通道转化为rgb三通道
#    save_path = "D:\\contrastive-predictive-coding-master\\images\\zhuan"
#    img.save(save_path + image)

from PIL import Image
import os
path = "D:/contrastive-predictive-coding-master/chl/"
all_images = os.listdir(path)
    # print(all_images)
for image in all_images:
    image_path = os.path.join(path, image)
    img = Image.open(image_path)  # 打开图片
    img = img.convert("RGB")  # 4通道转化为rgb三通道
    save_path = 'D:/contrastive-predictive-coding-master/new/chl/'
    img.save(save_path + image)