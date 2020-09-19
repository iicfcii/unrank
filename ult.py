import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# All based on 1280*720
# Rects for player 8
ULT_DIGIT_1_LEFT = 913
ULT_DIGIT_1_TOP = 54
ULT_DIGIT_1_WIDTH = 6
ULT_DIGIT_1_HEIGHT = 15
ULT_DIGIT_1_TILT = 4
ULT_DIGIT_1_MASK_1_X_OFFSET = 3
ULT_DIGIT_1_MASK_7_X_OFFSET = 1
ULT_DIGIT_1_MASK_POLY = np.array((ULT_DIGIT_1_LEFT,ULT_DIGIT_1_TOP,
                                       ULT_DIGIT_1_LEFT+ULT_DIGIT_1_WIDTH,ULT_DIGIT_1_TOP,
                                       ULT_DIGIT_1_LEFT+ULT_DIGIT_1_WIDTH-ULT_DIGIT_1_TILT,ULT_DIGIT_1_TOP+ULT_DIGIT_1_HEIGHT,
                                       ULT_DIGIT_1_LEFT-ULT_DIGIT_1_TILT,ULT_DIGIT_1_TOP+ULT_DIGIT_1_HEIGHT),
                                       dtype=np.float32).reshape((-1,2))
ULT_DIGIT_1_RECT = cv2.boundingRect(ULT_DIGIT_1_MASK_POLY)
ULT_READY_RECT = (904,54,16,16)

ULT_RECT_7 = (830,50,24,24)
ULT_RECT_1 = (35,50,24,24)
RECT_X_OFFSET = 70.5

ULT_BG_INDEX = 11610
ULT_BG_RECT_1 = (35,50,24,4)
ULT_BG_RECT_7 = (830,50,24,4)

MATCH_THRESHOLD = 0.9
RATIO = 3
ULT_NUMBER_TOO_CLOSE_THRESHOLD = 5*RATIO

NUMBER_TEMPLATES = (0,1,2,3,4,5,6,7,8,9,100)
NUMBER_IMAGE_INDICES = (2340, 2190, 2220, 2580, 3000, 2610, 3210, 3300, 3450, 3570, 11160) # (0, 1,..., 9, 100)

def crop(img, rect):
    img_crop = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    return img_crop

def read_ult_rects():
    ult_rects = {}

    for i in range(1,7):
        ult_rects[i] = (int(ULT_RECT_1[0]+(i-1)*RECT_X_OFFSET),
                         ULT_RECT_1[1],
                         ULT_RECT_1[2],
                         ULT_RECT_1[3])

    for i in range(7,13):
        ult_rects[i] = (int(ULT_RECT_7[0]+(i-7)*RECT_X_OFFSET),
                         ULT_RECT_7[1],
                         ULT_RECT_7[2],
                         ULT_RECT_7[3])

    # img = cv2.imread('img/overwatch_1_1_2340.jpg', cv2.IMREAD_COLOR)
    # for i in range(1,13):
    #     img_mark = cv2.rectangle(img, ult_rects[i], (255,255,255), thickness=1)
    # cv2.imshow('img', img_mark)
    # cv2.waitKey(0)
    return ult_rects

def read_ult_bg_rects(ult_rects):
    ult_bg_rects = {}

    for i in range(1,7):
        ult_bg_rects[i] = (
            ult_rects[i][0]+ULT_BG_RECT_1[0]-ULT_RECT_1[0],
            ult_rects[i][1]+ULT_BG_RECT_1[1]-ULT_RECT_1[1],
            ULT_BG_RECT_1[2],
            ULT_BG_RECT_1[3]
        )

    for i in range(7,13):
        ult_bg_rects[i] = (
            ult_rects[i][0]+ULT_BG_RECT_7[0]-ULT_RECT_7[0],
            ult_rects[i][1]+ULT_BG_RECT_7[1]-ULT_RECT_7[1],
            ULT_BG_RECT_7[2],
            ULT_BG_RECT_7[3]
        )

    # img = cv2.imread('img/overwatch_1_1_2340.jpg', cv2.IMREAD_COLOR)
    # for i in range(1,13):
    #     img_mark = cv2.rectangle(img, ult_bg_rects[i], (255,255,255), thickness=1)
    # cv2.imshow('img', img_mark)
    # cv2.waitKey(0)
    return ult_bg_rects

def save_templates():
    for number, index in enumerate(NUMBER_IMAGE_INDICES):
        img = cv2.imread('img/overwatch_1_1_'+str(index)+'.jpg', cv2.IMREAD_COLOR)
        if number < 10:
            img_mark = cv2.polylines(img.copy(), [ULT_DIGIT_1_MASK_POLY.astype(np.int32)], True, (255,255,255), thickness=1)
            cv2.imwrite('template/ult_number_'+str(number)+'.jpg', crop(img, ULT_DIGIT_1_RECT))
        else:
            img_mark = cv2.rectangle(img.copy(), ULT_READY_RECT, (255,255,255), thickness=1)
            cv2.imwrite('template/ult_number_100.jpg', crop(img, ULT_READY_RECT))
        # cv2.imshow('img', img_mark)
        # cv2.waitKey(0)

def read_number_template(number):
    img = cv2.imread('template/ult_number_'+str(number)+'.jpg', cv2.IMREAD_GRAYSCALE) # IMREAD_COLOR

    if number < 10:
        mask_poly_offset = ULT_DIGIT_1_MASK_POLY-np.tile(ULT_DIGIT_1_RECT[0:2],(4,1))

        # Adjust mask a bit for 1 and 7
        if number == 1:
            mask_poly_offset[0,0] = mask_poly_offset[0,0]+ULT_DIGIT_1_MASK_1_X_OFFSET
            mask_poly_offset[3,0] = mask_poly_offset[3,0]+ULT_DIGIT_1_MASK_1_X_OFFSET

        if number == 7:
            mask_poly_offset[0,0] = mask_poly_offset[0,0]+ULT_DIGIT_1_MASK_7_X_OFFSET
            mask_poly_offset[3,0] = mask_poly_offset[3,0]+ULT_DIGIT_1_MASK_7_X_OFFSET

        img_mask = np.zeros(img.shape[0:2], dtype=np.uint8)
        cv2.fillConvexPoly(img_mask, mask_poly_offset.astype(np.int32), (255,255,255))
    else:
        img_mask = np.zeros(img.shape[0:2], dtype=np.uint8)
        img_mask[:,:] = 255
    return (img, img_mask)

def read_templates(ratio):
    templates = {}
    for number in NUMBER_TEMPLATES:
        template, mask = read_number_template(number)
        template[mask == 0] = 0 # Note: cover other area so that number dominates difference

        template = cv2.resize(template, None, fx=ratio, fy=ratio)
        mask = cv2.resize(mask, None, fx=ratio, fy=ratio)

        templates[number] = (template, mask)
        # cv2.imshow(str(number), template)
        # cv2.waitKey(0)

    return templates

# Assume image should be in bgr format
def read_ult(img, ult_rect, ratio, templates):
    img_cropped = crop(img, ult_rect)
    img_cropped = cv2.resize(img_cropped, None, fx=ratio, fy=ratio)
    img_cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

    match = []
    for number in NUMBER_TEMPLATES:
        template, mask = templates[number]
        # cv2.imshow('template', template)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)

        res = cv2.matchTemplate(img_cropped_gray, template, cv2.TM_CCOEFF_NORMED, mask=mask)

        # print(np.where(res>=0.8))
        locations = np.where(res>=MATCH_THRESHOLD)

        if len(locations[0]) == 0: continue

        # Sort locations so that ones with higher res are preserved
        locations = [(locations[1][i], locations[0][i]) for i in range(len(locations[0]))]
        locations.sort(reverse=True, key=lambda l:res[l[1],l[0]])

        # Clean up clustered location
        locations_simple = []
        for l in locations:
            tooClose = False
            for l_simple in locations_simple:
                if (l[0]-l_simple[0])**2+(l[1]-l_simple[1])**2 < ULT_NUMBER_TOO_CLOSE_THRESHOLD **2:
                    tooClose = True
                    break
            if not tooClose:
                locations_simple.append(l)
                match.append((number,l, res[l[1],l[0]])) # Consider as two numbers

    if len(match) > 2:
        # Only save two matches with highest res
        match.sort(reverse=True, key=lambda l:l[2])
        match = match[0:2]

    for i in range(len(match)-1,-1,-1):
        # Remove match with inf res
        if match[i][2] == float('inf'):
            match.pop(i)

    # Sort according to x so that value is correct
    match.sort(key=lambda l:l[1][0])

    # Calculate ult value
    if len(match) == 2:
        ult = match[0][0]*10+match[1][0]
    elif len(match) == 1:
        ult = match[0][0]
    else:
        ult = -1

    # print(ult, match)
    # cv2.imshow('gray', img_cropped_gray)
    # cv2.waitKey(0)

    return ult

def read_ults(img, ult_rects, ratio, templates):
    ults = {}
    for key in ult_rects:
        ults[key] = read_ult(img, ult_rects[key], ratio, templates)

    return ults

def read_match_ults(start, end):
    templates = read_templates(RATIO)
    ult_rects = read_ult_rects()

    data = {}
    data['ult'] = {}

    for key in ult_rects:
        data['ult'][key] = []

    count = 0
    for i in range(start, end):
        index = i*30
        # NOTE: Read image takes long time
        img = cv2.imread('img/overwatch_1_1_'+str(index)+'.jpg', cv2.IMREAD_COLOR)
        ults = read_ults(img, ult_rects, RATIO, templates)

        for key in ults:
            data['ult'][key].append(ults[key])

        count += 1
        print('{:.0f}%'.format(100*count/(end-start)))

    return data

def save_match_ults():
    data = read_match_ults(0, 732)

    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)

def plot_match_ults():
    with open('data.json') as json_file:
        data = json.load(json_file)
    with open('data_ults.json') as json_file:
        data_old = json.load(json_file)

    end = 13
    for i in range(1,end):
        plt.subplot(end-1,1,i)
        plt.plot(data_old['ult'][str(i)])
        plt.plot(data['ult'][str(i)])
    plt.show()

def shift_img(img, mean_ref):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    mean = (np.mean(img_lab[:,:,0]), np.mean(img_lab[:,:,1]), np.mean(img_lab[:,:,2]))

    img_lab = img_lab.astype(np.float)
    img_lab[:,:,0] = img_lab[:,:,0]-mean[0]+mean_ref[0]
    img_lab[:,:,1] = img_lab[:,:,1]-mean[1]+mean_ref[1]
    img_lab[:,:,2] = img_lab[:,:,2]-mean[2]+mean_ref[2]
    img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)

    return cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)

def remove_spikes(ult, window_size):
    i = 0
    while(i+window_size < len(ult)):
        delta_ult = ult[i+1:i+window_size]-ult[i:i+window_size-1]
        change_total = np.sum(np.absolute(delta_ult))
        change_net = np.absolute(np.sum(delta_ult))
        if change_net < change_total*0.1:
            ult[i+1:i+window_size-1] = (ult[i]+ult[i+window_size-1])/2
        i += 1

def test_bg():
    templates = read_templates(RATIO)
    ult_rects = read_ult_rects()
    ult_bg_rects = read_ult_bg_rects(ult_rects)

    with open('data_ults.json') as json_file:
        data = json.load(json_file)

    player = 7
    img_ref = cv2.imread('template/ult_background_7.jpg', cv2.IMREAD_COLOR)
    img_ref_lab = cv2.cvtColor(img_ref, cv2.COLOR_BGR2Lab)
    mean_ref = (np.mean(img_ref_lab[:,:,0]), np.mean(img_ref_lab[:,:,1]), np.mean(img_ref_lab[:,:,2]))
    # cv2.imshow('ref', img_ref)
    # cv2.waitKey(0)

    ult_shifted = data['ult'][str(player)].copy()

    mean_diff = []
    for i in range(0,732):
        img_src = cv2.imread('img/overwatch_1_1_'+str(i*30)+'.jpg', cv2.IMREAD_COLOR)
        img = crop(img_src, ult_bg_rects[player])
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        mean = (np.mean(img_lab[:,:,0]), np.mean(img_lab[:,:,1]), np.mean(img_lab[:,:,2]))
        diff = np.sqrt((mean_ref[0]-mean[0])**2+(mean_ref[1]-mean[1])**2+(mean_ref[2]-mean[2])**2)
        mean_diff.append(diff)

        if diff > 40:
            ult = read_ult(shift_img(img_src, mean_ref), ult_rects[player], RATIO, templates)
            ult_shifted[i] = ult
            # cv2.imshow('ref', img_ref)
            # cv2.imshow('before', crop(img_src, ult_rects[player]))
            # cv2.imshow('after', shift_img(crop(img_src, ult_rects[player]), mean_ref))
            # cv2.waitKey(0)

    plt.subplot(3,1,1)
    plt.plot(mean_diff)
    plt.subplot(3,1,2)
    plt.plot(data['ult'][str(player)])
    plt.subplot(3,1,3)
    plt.plot(ult_shifted)
    plt.show()

def test_read_ult():
    index = 95
    player = 7
    img_src = cv2.imread('img/overwatch_1_1_'+str(index*30)+'.jpg', cv2.IMREAD_COLOR)

    templates = read_templates(RATIO)
    ult_rects = read_ult_rects()

    ult = read_ult(img_src, ult_rects[player], RATIO, templates)

    print(ult)
    cv2.imshow('src', img_src)
    cv2.waitKey(0)

def test_remove_spikes():
    with open('data_ults.json') as json_file:
        data = json.load(json_file)

    for player in range(1,13):
        ult_src = np.array(data['ult'][str(player)])
        ult = ult_src.copy()

        remove_spikes(ult, 6)
        plt.subplot(12,1,player)
        plt.plot(ult_src)
        plt.plot(ult)

    plt.show()

def test_extract_ult():
    ult_rects = read_ult_rects()
    ult_bg_rects = read_ult_bg_rects(ult_rects)

    img_src = cv2.imread('img/overwatch_1_1_15720.jpg', cv2.IMREAD_COLOR)

    player = 1
    img_ult = crop(img_src, ult_rects[player])
    img_ult_bg = crop(img_src, ult_bg_rects[player])

    mean_bg = np.array([np.mean(img_ult_bg[:,:,0]), np.mean(img_ult_bg[:,:,1]), np.mean(img_ult_bg[:,:,2])])
    std_bg = np.array([np.std(img_ult_bg[:,:,0]), np.std(img_ult_bg[:,:,1]), np.std(img_ult_bg[:,:,2])])
    lb = mean_bg-5*std_bg
    ub = mean_bg+5*std_bg

    print(mean_bg, std_bg, lb, ub)

    img_ult_mask = cv2.inRange(img_ult, lb, ub)

    img_ult_mask_tmp = np.zeros(img_ult_mask.shape, dtype=np.uint8)
    img_ult_mask_tmp[img_ult_mask == 0] = 255
    img_ult_mask_tmp = cv2.erode(img_ult_mask_tmp,cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations=1)
    img_ult[img_ult_mask_tmp == 0] = 0

    cv2.imshow('ult', img_ult)
    cv2.imshow('ult bg', img_ult_bg)
    cv2.imshow('ult mask', img_ult_mask_tmp)
    cv2.waitKey(0)

# test_bg()
# test_read_ult()
# save_match_ults()
# plot_match_ults()
# test_remove_spikes()
test_extract_ult()
