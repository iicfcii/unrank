import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Rects for player 8
STAT_ULT_DIGIT_1_LEFT = 913
STAT_ULT_DIGIT_1_TOP = 54
STAT_ULT_DIGIT_1_WIDTH = 6
STAT_ULT_DIGIT_1_HEIGHT = 15
STAT_ULT_DIGIT_1_TILT = 4
STAT_ULT_DIGIT_1_MASK_1_X_OFFSET = 3
STAT_ULT_DIGIT_1_MASK_7_X_OFFSET = 1
STAT_ULT_DIGIT_1_MASK_POLY = np.array((STAT_ULT_DIGIT_1_LEFT,STAT_ULT_DIGIT_1_TOP,
                                       STAT_ULT_DIGIT_1_LEFT+STAT_ULT_DIGIT_1_WIDTH,STAT_ULT_DIGIT_1_TOP,
                                       STAT_ULT_DIGIT_1_LEFT+STAT_ULT_DIGIT_1_WIDTH-STAT_ULT_DIGIT_1_TILT,STAT_ULT_DIGIT_1_TOP+STAT_ULT_DIGIT_1_HEIGHT,
                                       STAT_ULT_DIGIT_1_LEFT-STAT_ULT_DIGIT_1_TILT,STAT_ULT_DIGIT_1_TOP+STAT_ULT_DIGIT_1_HEIGHT),
                                       dtype=np.float32).reshape((-1,2))
STAT_ULT_DIGIT_1_RECT = cv2.boundingRect(STAT_ULT_DIGIT_1_MASK_POLY)
STAT_ULT_READY_RECT = (904,54,16,16)

# Based on 1280*720
# STAT_ULT_RECT_7 = (825,48,81,53)
# STAT_ULT_RECT_1 = (22,48,81,53)
STAT_ULT_RECT_7 = (830,50,24,24)
STAT_ULT_RECT_1 = (35,50,24,24)
STAT_RECT_X_OFFSET = 70.5

STAT_ULT_BACKGROUND_COLOR_INDEX = 11610
STAT_ULT_BACKGROUND_COLOR_RECT_1 = (35,50,24,4)
STAT_ULT_BACKGROUND_COLOR_RECT_7 = (830,50,24,4)

THRESHOLD = 0.9
RATIO = 3
STAT_ULT_NUMBER_TOO_CLOSE_THRESHOLD = 5*RATIO

NUMBER_TEMPLATES = (0,1,2,3,4,5,6,7,8,9,100)
NUMBER_IMAGE_INDICES = (2340, 2190, 2220, 2580, 3000, 2610, 3210, 3300, 3450, 3570, 11160) # (0, 1,..., 9, 100)

def crop(img, rect):
    img_crop = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    return img_crop

def read_stat_ult_rects():
    stat_ult_rects = {}

    for i in range(1,7):
        stat_ult_rects[i] = (int(STAT_ULT_RECT_1[0]+(i-1)*STAT_RECT_X_OFFSET),
                         STAT_ULT_RECT_1[1],
                         STAT_ULT_RECT_1[2],
                         STAT_ULT_RECT_1[3])

    for i in range(7,13):
        stat_ult_rects[i] = (int(STAT_ULT_RECT_7[0]+(i-7)*STAT_RECT_X_OFFSET),
                         STAT_ULT_RECT_7[1],
                         STAT_ULT_RECT_7[2],
                         STAT_ULT_RECT_7[3])

    # img = cv2.imread('img/overwatch_1_1_2340.jpg', cv2.IMREAD_COLOR)
    # for i in range(1,13):
    #     img_mark = cv2.rectangle(img, stat_ult_rects[i], (255,255,255), thickness=1)
    # cv2.imshow('img', img_mark)
    # cv2.waitKey(0)
    return stat_ult_rects

def read_stat_ult_background_color_rects(stat_ult_rects):
    stat_ult_background_color_rects = {}

    for i in range(1,7):
        stat_ult_background_color_rects[i] = (
            stat_ult_rects[i][0]+STAT_ULT_BACKGROUND_COLOR_RECT_1[0]-STAT_ULT_RECT_1[0],
            stat_ult_rects[i][1]+STAT_ULT_BACKGROUND_COLOR_RECT_1[1]-STAT_ULT_RECT_1[1],
            STAT_ULT_BACKGROUND_COLOR_RECT_1[2],
            STAT_ULT_BACKGROUND_COLOR_RECT_1[3]
        )

    for i in range(7,13):
        stat_ult_background_color_rects[i] = (
            stat_ult_rects[i][0]+STAT_ULT_BACKGROUND_COLOR_RECT_7[0]-STAT_ULT_RECT_7[0],
            stat_ult_rects[i][1]+STAT_ULT_BACKGROUND_COLOR_RECT_7[1]-STAT_ULT_RECT_7[1],
            STAT_ULT_BACKGROUND_COLOR_RECT_7[2],
            STAT_ULT_BACKGROUND_COLOR_RECT_7[3]
        )

    # img = cv2.imread('img/overwatch_1_1_2340.jpg', cv2.IMREAD_COLOR)
    # for i in range(1,13):
    #     img_mark = cv2.rectangle(img, stat_ult_background_color_rects[i], (255,255,255), thickness=1)
    # cv2.imshow('img', img_mark)
    # cv2.waitKey(0)
    return stat_ult_background_color_rects

def save_templates():
    for number, index in enumerate(NUMBER_IMAGE_INDICES):
        img = cv2.imread('img/overwatch_1_1_'+str(index)+'.jpg', cv2.IMREAD_COLOR)
        if number < 10:
            img_mark = cv2.polylines(img.copy(), [STAT_ULT_DIGIT_1_MASK_POLY.astype(np.int32)], True, (255,255,255), thickness=1)
            cv2.imwrite('template/ult_number_'+str(number)+'.jpg', crop(img, STAT_ULT_DIGIT_1_RECT))
        else:
            img_mark = cv2.rectangle(img.copy(), STAT_ULT_READY_RECT, (255,255,255), thickness=1)
            cv2.imwrite('template/ult_number_100.jpg', crop(img, STAT_ULT_READY_RECT))
        # cv2.imshow('img', img_mark)
        # cv2.waitKey(0)

    # img = cv2.imread('img/overwatch_1_1_'+str(STAT_ULT_BACKGROUND_COLOR_INDEX)+'.jpg', cv2.IMREAD_COLOR)
    # img = crop(img, STAT_ULT_BACKGROUND_COLOR_RECT_1)
    # cv2.imwrite('template/ult_background_1.jpg', img)
    #
    # img = cv2.imread('img/overwatch_1_1_'+str(STAT_ULT_BACKGROUND_COLOR_INDEX)+'.jpg', cv2.IMREAD_COLOR)
    # img = crop(img, STAT_ULT_BACKGROUND_COLOR_RECT_7)
    # cv2.imwrite('template/ult_background_7.jpg', img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

def read_number_template(number):
    img = cv2.imread('template/ult_number_'+str(number)+'.jpg', cv2.IMREAD_GRAYSCALE) # IMREAD_COLOR

    if number < 10:
        mask_poly_offset = STAT_ULT_DIGIT_1_MASK_POLY-np.tile(STAT_ULT_DIGIT_1_RECT[0:2],(4,1))

        # Adjust mask a bit for 1 and 7
        if number == 1:
            mask_poly_offset[0,0] = mask_poly_offset[0,0]+STAT_ULT_DIGIT_1_MASK_1_X_OFFSET
            mask_poly_offset[3,0] = mask_poly_offset[3,0]+STAT_ULT_DIGIT_1_MASK_1_X_OFFSET

        if number == 7:
            mask_poly_offset[0,0] = mask_poly_offset[0,0]+STAT_ULT_DIGIT_1_MASK_7_X_OFFSET
            mask_poly_offset[3,0] = mask_poly_offset[3,0]+STAT_ULT_DIGIT_1_MASK_7_X_OFFSET

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
def read_ult(img, stat_ult_rect, ratio, templates):
    img_cropped = crop(img, stat_ult_rect)
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
        locations = np.where(res>=THRESHOLD)

        if len(locations[0]) == 0: continue

        # Sort locations so that ones with higher res are preserved
        locations = [(locations[1][i], locations[0][i]) for i in range(len(locations[0]))]
        locations.sort(reverse=True, key=lambda l:res[l[1],l[0]])

        # Clean up clustered location
        locations_simple = []
        for l in locations:
            tooClose = False
            for l_simple in locations_simple:
                if (l[0]-l_simple[0])**2+(l[1]-l_simple[1])**2 < STAT_ULT_NUMBER_TOO_CLOSE_THRESHOLD **2:
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

def read_ults(img, stat_ult_rects, ratio, templates):
    ults = {}
    for key in stat_ult_rects:
        ults[key] = read_ult(img, stat_ult_rects[key], ratio, templates)

    return ults

def read_match_ults(start, end):
    templates = read_templates(RATIO)
    stat_ult_rects = read_stat_ult_rects()

    data = {}
    data['ult'] = {}

    for key in stat_ult_rects:
        data['ult'][key] = []

    count = 0
    for i in range(start, end):
        index = i*30
        # NOTE: Read image takes long time
        img = cv2.imread('img/overwatch_1_1_'+str(index)+'.jpg', cv2.IMREAD_COLOR)
        ults = read_ults(img, stat_ult_rects, RATIO, templates)

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
    # std = (np.std(img_lab[:,:,0]), np.std(img_lab[:,:,1]), np.std(img_lab[:,:,2]))

    img_lab = img_lab.astype(np.float)
    img_lab[:,:,0] = img_lab[:,:,0]-mean[0]+mean_ref[0]
    img_lab[:,:,1] = img_lab[:,:,1]-mean[1]+mean_ref[1]
    img_lab[:,:,2] = img_lab[:,:,2]-mean[2]+mean_ref[2]
    img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)

    return cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)

def test_background_color():
    templates = read_templates(RATIO)
    stat_ult_rects = read_stat_ult_rects()
    stat_ult_background_color_rects = read_stat_ult_background_color_rects(stat_ult_rects)

    with open('data_ults.json') as json_file:
        data = json.load(json_file)

    player = 7
    img_ref = cv2.imread('template/ult_background_7.jpg', cv2.IMREAD_COLOR)
    img_ref_lab = cv2.cvtColor(img_ref, cv2.COLOR_BGR2Lab)
    mean_ref = (np.mean(img_ref_lab[:,:,0]), np.mean(img_ref_lab[:,:,1]), np.mean(img_ref_lab[:,:,2]))
    # stf_ref = (np.std(img_ref_lab[:,:,0]), np.std(img_ref_lab[:,:,1]), np.std(img_ref_lab[:,:,2]))
    # cv2.imshow('ref', img_ref)
    # cv2.waitKey(0)

    ult_shifted = data['ult'][str(player)].copy()

    mean_diff = []
    for i in range(0,732):
        img_src = cv2.imread('img/overwatch_1_1_'+str(i*30)+'.jpg', cv2.IMREAD_COLOR)
        img = crop(img_src, stat_ult_background_color_rects[player])
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        mean = (np.mean(img_lab[:,:,0]), np.mean(img_lab[:,:,1]), np.mean(img_lab[:,:,2]))
        diff = np.sqrt((mean_ref[0]-mean[0])**2+(mean_ref[1]-mean[1])**2+(mean_ref[2]-mean[2])**2)
        mean_diff.append(diff)

        if diff > 40:
            ult = read_ult(shift_img(img_src, mean_ref), stat_ult_rects[player], RATIO, templates)
            ult_shifted[i] = ult
            # cv2.imshow('ref', img_ref)
            # cv2.imshow('before', crop(img_src, stat_ult_rects[player]))
            # cv2.imshow('after', shift_img(crop(img_src, stat_ult_rects[player]), mean_ref))
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
    stat_ult_rects = read_stat_ult_rects()

    ult = read_ult(img_src, stat_ult_rects[player], RATIO, templates)

    print(ult)
    cv2.imshow('src', img_src)
    cv2.waitKey(0)


def find_invalid():
    with open('data_ults.json') as json_file:
        data = json.load(json_file)

    player = 1
    ult = data['ult'][str(player)]

    window_size = 3

    for i in range(len(ult)-window_size):
        pass

# test_background_color()
# test_read_ult()
# save_match_ults()
plot_match_ults()
