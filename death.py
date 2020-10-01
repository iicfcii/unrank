import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

ELIMINATED_RECT_1 = (45,90,20,14)
ELIMINATED_RECT_7 = (860,90,20,14)
STAT_RECT_X_OFFSET = 71
MATCH_PADDING = 2

ELIMINATED_COLOR = np.array((175.0, 200.0, 180.0)) #HSV
ELIMINATED_COLOR_RANGE = np.array((20.0, 50.0, 70.0))
ELIMINATED_COLOR_LB = ELIMINATED_COLOR-ELIMINATED_COLOR_RANGE
ELIMINATED_COLOR_UB = ELIMINATED_COLOR+ELIMINATED_COLOR_RANGE

def crop(img, rect):
    x, y, w, h = rect

    if x < 0: x = 0
    if y < 0: y = 0
    if x+w > img.shape[1]: w = img.shape[1]-x
    if y+h > img.shape[0]: h = img.shape[0]-y

    img_crop = img[y:y+h,x:x+w]
    return img_crop

def read_elim_rects():
    elim_rects = {}

    for i in range(1,7):
        elim_rects[i] = (int(ELIMINATED_RECT_1[0]+(i-1)*STAT_RECT_X_OFFSET),
                                ELIMINATED_RECT_1[1],
                                ELIMINATED_RECT_1[2],
                                ELIMINATED_RECT_1[3])

    for i in range(7,13):
        elim_rects[i] = (int(ELIMINATED_RECT_7[0]+(i-7)*STAT_RECT_X_OFFSET),
                                ELIMINATED_RECT_7[1],
                                ELIMINATED_RECT_7[2],
                                ELIMINATED_RECT_7[3])

    # img = cv2.imread('img/overwatch_1_1_17400.jpg', cv2.IMREAD_COLOR) # 3030
    # for i in range(1,13):
    #     img_mark = cv2.rectangle(img, elim_rects[i], (255,255,255), thickness=1)
    # cv2.imshow('img', img_mark)
    # cv2.waitKey(0)

    return elim_rects

def save_elim_templates():
    elim_rects = read_elim_rects()
    img = cv2.imread('img/overwatch_1_1_3060.jpg', cv2.IMREAD_COLOR)
    img_mark = cv2.rectangle(img.copy(), elim_rects[3], (255,255,255), thickness=1)
    cv2.imwrite('template/health_eliminated_1.jpg', crop(img, elim_rects[3]))

    img = cv2.imread('img/overwatch_1_1_15570.jpg', cv2.IMREAD_COLOR)
    img_mark = cv2.rectangle(img.copy(), elim_rects[12], (255,255,255), thickness=1)
    cv2.imwrite('template/health_eliminated_2.jpg', crop(img, elim_rects[12]))

    # cv2.imshow('img', img_mark)
    # cv2.waitKey(0)

def read_elim_templates():
    img_1 = cv2.imread('template/health_eliminated_1.jpg', cv2.IMREAD_COLOR) # IMREAD_COLOR
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    ret, img_bin_1 = cv2.threshold(img_1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    img_2 = cv2.imread('template/health_eliminated_2.jpg', cv2.IMREAD_COLOR) # IMREAD_COLOR
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    ret, img_bin_2 = cv2.threshold(img_2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # cv2.imshow('img', img_bin_2)
    # cv2.waitKey(0)

    return img_bin_1, img_bin_2

def match_color(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_bin = np.zeros((img.shape[0],img.shape[1],1), dtype=np.uint8)

    if ELIMINATED_COLOR_UB[0] > 180: # Red is across 180
        h_match = np.logical_or(img_hsv[:,:,0] > ELIMINATED_COLOR_LB[0], img_hsv[:,:,0] < ELIMINATED_COLOR_UB[0]-180)
    else:
        h_match = np.logical_and(img_hsv[:,:,0] > ELIMINATED_COLOR_LB[0], img_hsv[:,:,0] < ELIMINATED_COLOR_UB[0])
    s_match = np.logical_and(img_hsv[:,:,1] > ELIMINATED_COLOR_LB[1], img_hsv[:,:,1] < ELIMINATED_COLOR_UB[1])
    v_match = np.logical_and(img_hsv[:,:,2] > ELIMINATED_COLOR_LB[2], img_hsv[:,:,2] < ELIMINATED_COLOR_UB[2])

    img_bin[np.all((h_match, s_match, v_match), axis=0)] = 255

    return img_bin

def read_elim(img, elim_rect, templates):
    elim_rect = (elim_rect[0]-MATCH_PADDING,
                 elim_rect[1]-MATCH_PADDING,
                 elim_rect[2]+2*MATCH_PADDING,
                 elim_rect[3]+2*MATCH_PADDING)

    img = crop(img, elim_rect)

    img_bin = match_color(img)
    img[np.tile(img_bin > 0, (1,1,3))] = 255

    temp_1, temp_2 = templates
    if elim_rect[0] < ELIMINATED_RECT_7[0]:
        temp = temp_1
    else:
        temp = temp_2

    res = cv2.matchTemplate(img_bin, temp, cv2.TM_CCOEFF_NORMED)

    # print(res)
    # cv2.imshow('img', img)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    score = np.max(res)
    if score > 0.5:
        return 1, img
    else:
        return 0, img

def read_elims(img, elim_rects, templates):
    elims = {}

    for p in elim_rects:
        elim, img_bin = read_elim(img, elim_rects[p], templates)
        elims[p] = elim

    return elims

def read_match_elims(start, end):
    templates = read_elim_templates()
    elim_rects = read_elim_rects()

    data = {'elims':{}}
    for p in elim_rects:
        data['elims'][p] = []

    count = 0
    for i in range(start, end):
        img = cv2.imread('img/overwatch_1_1_'+str(i*30)+'.jpg', cv2.IMREAD_COLOR)
        elims = read_elims(img, elim_rects, templates)

        for p in elims:
            data['elims'][p].append(elims[p])

        count += 1
        print('{:.0f}%'.format(100*count/(end-start)))

    return data

def save_match_elims():
    data = read_match_elims(0, 732)

    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)

def plot_match_elims():
    with open('data.json') as json_file:
        data = json.load(json_file)

    end = 13
    for i in range(1,end):
        plt.subplot(end-1,1,i)
        plt.plot(data['elims'][str(i)])
    plt.show()

def remove_spikes(ult, window_size):
    i = 0
    while(i+window_size < len(ult)):
        delta_ult = ult[i+1:i+window_size]-ult[i:i+window_size-1]
        change_total = np.sum(np.absolute(delta_ult))
        change_net = np.absolute(np.sum(delta_ult))
        # Only detect downward spike
        if change_net < change_total*0.1 and ult[i] != 0:
            ult[i+1:i+window_size-1] = (ult[i]+ult[i+window_size-1])/2
        i += 1

def test_remove_spikes():
    with open('data.json') as json_file:
        data = json.load(json_file)

    for player in range(1,13):
        ult_src = np.array(data['elims'][str(player)])
        ult = ult_src.copy()

        remove_spikes(ult, 3)
        plt.subplot(12,1,player)
        plt.plot(ult_src)
        plt.plot(ult)

    plt.show()

def read_batch(num_width=15, num_height=8):
    templates = read_elim_templates()
    elim_rects = read_elim_rects()

    for i in range(0,int(732/num_width/num_height)+1):
        imgs = []
        for j in range(i*num_width*num_height, i*num_width*num_height+num_width*num_height):
            img = cv2.imread('img/overwatch_1_1_'+str(j*30)+'.jpg', cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((ELIMINATED_RECT_1[3]+MATCH_PADDING*2, ELIMINATED_RECT_1[2]+MATCH_PADDING*2, 3), dtype=np.uint8)
                elim = 0
            else:
                # print(j*30)
                elim, img = read_elim(img, elim_rects[7], templates)


            if j < 732: img = cv2.putText(img, str(elim), (0,img.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            imgs.append(img)

        imgs_row = []
        for i in range(num_height):
            imgs_row.append(cv2.hconcat(imgs[i*num_width:i*num_width+num_width], num_width))

        img_final = cv2.vconcat(imgs_row, num_height)
        cv2.imshow('read', img_final)
        cv2.waitKey(0)


# save_elim_templates()
# templates = read_elim_templates()
# elim_rects = read_elim_rects()
# save_match_elims()
# plot_match_elims()
test_remove_spikes()
# read_batch(num_width=20, num_height=20)
