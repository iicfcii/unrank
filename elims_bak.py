import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# ELIMS_RECT_1 = (1070,110,190,175)
ELIMS_RECT = (910,110,350,175)

TEAM_1_COLOR = np.array((90, 0, 255)) #HSV
TEAM_1_COLOR_RANGE = np.array((90, 40, 40))
TEAM_1_COLOR_LB = TEAM_1_COLOR-TEAM_1_COLOR_RANGE
TEAM_1_COLOR_UB = TEAM_1_COLOR+TEAM_1_COLOR_RANGE


def crop(img, rect):
    x, y, w, h = rect

    if x < 0: x = 0
    if y < 0: y = 0
    if x+w > img.shape[1]: w = img.shape[1]-x
    if y+h > img.shape[0]: h = img.shape[0]-y

    img_crop = img[y:y+h,x:x+w]
    return img_crop

def save_elim_templates():
    img = cv2.imread('img/overwatch_1_1_20850.jpg', cv2.IMREAD_COLOR)

    rect = (1079,113,40,29)
    img_mark = cv2.rectangle(img.copy(), rect, (255,255,255), thickness=1)
    cv2.imwrite('template/elim_2.jpg', crop(img, rect))
    cv2.imshow('img', img_mark)
    cv2.waitKey(0)

    rect = (1146,113,40,29)
    img_mark = cv2.rectangle(img.copy(), rect, (255,255,255), thickness=1)
    cv2.imwrite('template/elim_1.jpg', crop(img, rect))
    cv2.imshow('img', img_mark)
    cv2.waitKey(0)

def read_elim_templates():
    img_1 = cv2.imread('template/elim_1.jpg', cv2.IMREAD_COLOR)
    img_2 = cv2.imread('template/elim_2.jpg', cv2.IMREAD_COLOR)
    h, w = img_1.shape[0:2]
    d = 3 # thickness
    mask_rect = np.array([[d,d],[w-d-1,d],[w-d-1,h-d-1],[d,h-d-1]],dtype=np.int32)

    mask = np.zeros((h,w), dtype=np.uint8)
    mask[:,:] = 255
    mask = cv2.fillConvexPoly(mask,mask_rect,0)

    img_1[mask == 0] = 0
    img_2[mask == 0] = 0

    cv2.imshow('elim 1', img_1)
    cv2.imshow('elim 2', img_2)
    cv2.waitKey(0)

    templates = {}
    templates['elim1'] = (img_1, mask)
    templates['elim2'] = (img_2, mask)

    return templates

def match_color(img, lb, ub):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_bin = np.zeros((img.shape[0],img.shape[1],1), dtype=np.uint8)

    if ub[0] > 180: # Red is across 180
        h_match = np.logical_or(img_hsv[:,:,0] > lb[0], img_hsv[:,:,0] < ub[0]-180)
    elif lb[0] < 0:
        h_match = np.logical_or(img_hsv[:,:,0] > lb[0]+180, img_hsv[:,:,0] < ub[0])
    else:
        h_match = np.logical_and(img_hsv[:,:,0] > lb[0], img_hsv[:,:,0] < ub[0])

    s_match = np.logical_and(img_hsv[:,:,1] > lb[1], img_hsv[:,:,1] < ub[1])
    v_match = np.logical_and(img_hsv[:,:,2] > lb[2], img_hsv[:,:,2] < ub[2])

    img_bin[np.all((h_match, s_match, v_match), axis=0)] = 255

    return img_bin

def extract_location(res):
    locations = np.where(res>=0.5)

    if len(locations[0]) == 0: return []

    # Sort locations so that ones with higher res are preserved
    # locations:[(x,y,score),(x,y,score),(x,y,score)...]
    locations = [(locations[1][i], locations[0][i], res[locations[0][i],locations[1][i]]) for i in range(len(locations[0]))]
    locations.sort(reverse=True, key=lambda l:l[2])

    # Clean up clustered location
    locations_simple = []
    for l in locations:
        tooClose = False
        for l_simple in locations_simple:
            if (l[0]-l_simple[0])**2+(l[1]-l_simple[1])**2 < 30 **2:
                tooClose = True
                break
        if not tooClose:
            locations_simple.append(l)

    # print(locations_simple)

    return locations_simple

def read_elim(img, templates):
    img = crop(img, ELIMS_RECT)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    template_1, mask_1 = templates['elim1']
    template_2, mask_2 = templates['elim2']
    # template_1 = cv2.cvtColor(template_1, cv2.COLOR_BGR2GRAY)
    # template_2 = cv2.cvtColor(template_2, cv2.COLOR_BGR2GRAY)

    res_1 = cv2.matchTemplate(img, template_1, cv2.TM_CCOEFF_NORMED, mask=mask_1)
    res_2 = cv2.matchTemplate(img, template_2, cv2.TM_CCOEFF_NORMED, mask=mask_2)
    res = np.maximum(res_1, res_2)

    locations = extract_location(res)

    for l in locations:
        img = cv2.rectangle(img, (l[0],l[1],template_1.shape[1],template_1.shape[0]), (255,255,255), thickness=2)

    return img

def read_batch(num_width=5, num_height=5):
    templates = read_elim_templates()

    for i in range(0,int(732/num_width/num_height)+1):
        imgs = []
        for j in range(i*num_width*num_height, i*num_width*num_height+num_width*num_height):
            img = cv2.imread('img/overwatch_1_1_'+str(j*30)+'.jpg', cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((ELIMS_RECT_1[3], ELIMS_RECT_1[2], 3), dtype=np.uint8)
                elim = 0
            else:
                # print(j*30)
                img = read_elim(img, templates)


            if j < 732: img = cv2.putText(img, str(j*30), (0,img.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            imgs.append(img)

        imgs_row = []
        for i in range(num_height):
            imgs_row.append(cv2.hconcat(imgs[i*num_width:i*num_width+num_width], num_width))

        img_final = cv2.vconcat(imgs_row, num_height)
        cv2.imshow('read', img_final)
        cv2.waitKey(0)

# save_elim_templates()
templates = read_elim_templates()

# img = cv2.imread('img/overwatch_1_1_20700.jpg', cv2.IMREAD_COLOR)
# img = read_elim(img, templates)

read_batch()
