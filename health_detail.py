import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

HEALTH_RECT = (170, 540, 170, 115)
HEALTH_NUMBER_WIDTH = 45
HEALTH_NUMBER_HEIGHT = 30

NAME_TAG_COLOR = (177, 243, 186) #HSV
NAME_TAG_H_RANGE = 40
NAME_TAG_S_RANGE = 100
NAME_TAG_V_RANGE = 100

h_lb = NAME_TAG_COLOR[0]-NAME_TAG_H_RANGE/2
s_lb = NAME_TAG_COLOR[1]-NAME_TAG_S_RANGE/2
v_lb = NAME_TAG_COLOR[2]-NAME_TAG_V_RANGE/2

h_ub = NAME_TAG_COLOR[0]+NAME_TAG_H_RANGE/2
s_ub = NAME_TAG_COLOR[1]+NAME_TAG_S_RANGE/2
v_ub = NAME_TAG_COLOR[2]+NAME_TAG_V_RANGE/2


if h_lb < 0:
    lb_sup = np.array([180+h_lb, s_lb, v_lb])
    ub_sup = np.array([180, s_ub, v_ub])
elif h_ub > 180:
    lb_sup = np.array([0, s_lb, v_lb])
    ub_sup = np.array([h_ub-180, s_ub, v_ub])
else:
    lb_sup = np.array([0, 0, 0])
    ub_sup = np.array([0, 0, 0])

lb = np.array([h_lb, s_lb, v_lb])
ub = np.array([h_ub, s_ub, v_ub])

def crop(img, rect):
    x, y, w, h = rect

    if x < 0: x = 0
    if y < 0: y = 0
    if x+w > img.shape[1]: w = img.shape[1]-x
    if y+h > img.shape[0]: h = img.shape[0]-y

    img_crop = img[y:y+h,x:x+w]
    return img_crop

def process(img):
    if img is None: return None

    img_src = crop(img, HEALTH_RECT)

    # Match color
    img = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img,(3,3),0)

    mask = cv2.inRange(img, lb, ub) # Black and white mask
    mask_sup = cv2.inRange(img, lb_sup, ub_sup)
    img = cv2.bitwise_or(mask, mask_sup)

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=8)

    # Remove small contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    img = np.zeros((img.shape[0],img.shape[1],1), dtype=np.uint8)
    if len(areas) > 0:
        max_index = np.argmax(areas)
        if areas[max_index] > 1000:
            # img = cv2.polylines(img, [contours[max_index]], True, 255)
            img = cv2.fillPoly(img, [contours[max_index]], 255)

    # Remove spikes on contours
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=10)

    x, y, w, h = cv2.boundingRect(img)
    if w == 0  or h == 0: return None

    rect_health = (x-15, y+h+2, HEALTH_NUMBER_WIDTH, HEALTH_NUMBER_HEIGHT)

    img = crop(img_src, rect_health)
    if img.shape[0] != HEALTH_NUMBER_HEIGHT or img.shape[1] != HEALTH_NUMBER_WIDTH: return None

    # img = cv2.rectangle(img_src, rect_health, 255)

    return img

def process_batch(num_width=15, num_height=8):
    for i in range(0,int(732/num_width/num_height)+1):
        imgs = []
        for j in range(i*num_width*num_height, i*num_width*num_height+num_width*num_height):
            img = cv2.imread('img/overwatch_1_1_'+str(j*30)+'.jpg', cv2.IMREAD_COLOR)
            img = process(img)

            if img is None:
                img = np.zeros((HEALTH_NUMBER_HEIGHT, HEALTH_NUMBER_WIDTH, 3), dtype=np.uint8)

            img = cv2.putText(img, str(j*30), (0,img.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0))
            imgs.append(img)

        imgs_row = []
        for i in range(num_height):
            imgs_row.append(cv2.hconcat(imgs[i*num_width:i*num_width+num_width], num_width))

        img_final = cv2.vconcat(imgs_row, num_height)
        cv2.imshow('edges', img_final)
        cv2.waitKey(0)

# print(cv2.cvtColor(np.array([[[210,42,29]]], dtype=np.uint8), cv2.COLOR_RGB2HSV))
# print(cv2.cvtColor(np.array([[[186,9,27]]], dtype=np.uint8), cv2.COLOR_RGB2HSV))
# print(cv2.cvtColor(np.array([[[189,21,21]]], dtype=np.uint8), cv2.COLOR_RGB2HSV))

process_batch(num_width=10, num_height=8)
