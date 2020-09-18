import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

STAT_ULT_LEFT = 913
STAT_ULT_TOP = 54
STAT_ULT_WIDTH = 6
STAT_ULT_HEIGHT = 15
STAT_ULT_TILT = 4
STAT_ULT_DIGIT_1_MASK_1_X_OFFSET = 2
STAT_ULT_DIGIT_1_MASK_7_X_OFFSET = 1
STAT_ULT_DIGIT_1_MASK_POLY = (STAT_ULT_LEFT,STAT_ULT_TOP,
                              STAT_ULT_LEFT+STAT_ULT_WIDTH,STAT_ULT_TOP,
                              STAT_ULT_LEFT+STAT_ULT_WIDTH-STAT_ULT_TILT,STAT_ULT_TOP+STAT_ULT_HEIGHT,
                              STAT_ULT_LEFT-STAT_ULT_TILT,STAT_ULT_TOP+STAT_ULT_HEIGHT)
mask_poly = np.array(STAT_ULT_DIGIT_1_MASK_POLY, dtype=np.float32).reshape((-1,2))
STAT_ULT_DIGIT_1_RECT = cv2.boundingRect(mask_poly)
STAT_ULT_READY_RECT = (904,54,16,16)

# Based on 1280*720
STAT_RECT_7 = (825,48,81,53)
STAT_RECT_1 = (22,48,81,53)
STAT_RECT_X_OFFSET = 70.5

THRESHOLD = 0.9
RATIO = 3
STAT_ULT_NUMBER_TOO_CLOSE_THRESHOLD = 5*RATIO

NUMBER_TEMPLATES = (0,1,2,3,4,5,6,7,8,9,100)
NUMBER_IMAGE_INDICES = (2340, 2190, 2220, 2580, 3000, 2610, 3210, 3300, 3450, 3570, 11160) # (0, 1,..., 9, 100)

def crop(img, rect):
    img_crop = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    return img_crop

def read_stat_rects():
    stat_rects = {}

    for i in range(1,7):
        stat_rects[i] = (int(STAT_RECT_1[0]+(i-1)*STAT_RECT_X_OFFSET),
                         STAT_RECT_1[1],
                         STAT_RECT_1[2],
                         STAT_RECT_1[3])

    for i in range(7,13):
        stat_rects[i] = (int(STAT_RECT_7[0]+(i-7)*STAT_RECT_X_OFFSET),
                         STAT_RECT_7[1],
                         STAT_RECT_7[2],
                         STAT_RECT_7[3])

    # img = cv2.imread('img/overwatch_1_1_2340.jpg', cv2.IMREAD_COLOR)
    # for i in range(1,13):
    #     img_mark = cv2.rectangle(img, stat_rects[i], (255,255,255), thickness=1)
    # cv2.imshow('img', img_mark)
    # cv2.waitKey(0)
    return stat_rects

def save_templates():
    for number, index in enumerate(NUMBER_IMAGE_INDICES):
        img = cv2.imread('img/overwatch_1_1_'+str(index)+'.jpg', cv2.IMREAD_COLOR)
        if number < 10:
            img_mark = cv2.polylines(img.copy(), [mask_poly.astype(np.int32)], True, (255,255,255), thickness=1)
            cv2.imwrite('template/ult_number_'+str(number)+'.jpg', crop(img, STAT_ULT_DIGIT_1_RECT))
        else:
            img_mark = cv2.rectangle(img.copy(), STAT_ULT_READY_RECT, (255,255,255), thickness=1)
            cv2.imwrite('template/ult_number_100.jpg', crop(img, STAT_ULT_READY_RECT))
        # cv2.imshow('img', img_mark)
        # cv2.waitKey(0)

def read_template(number):
    img = cv2.imread('template/ult_number_'+str(number)+'.jpg', cv2.IMREAD_GRAYSCALE) # IMREAD_COLOR

    if number < 10:
        mask_poly_offset = mask_poly-np.tile(STAT_ULT_DIGIT_1_RECT[0:2],(4,1))

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
        template, mask = read_template(number)
        template[mask == 0] = 0 # Note: cover other area so that number dominates difference

        template = cv2.resize(template, None, fx=ratio, fy=ratio)
        mask = cv2.resize(mask, None, fx=ratio, fy=ratio)

        templates[number] = (template, mask)
        # cv2.imshow(str(number), template)
        # cv2.waitKey(0)

    return templates

def read_ult(img, stat_rect, ratio, templates):
    img_cropped = crop(img, stat_rect)
    img_cropped = cv2.resize(img_cropped, None, fx=ratio, fy=ratio)
    img_cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

    match = []
    for number in NUMBER_TEMPLATES:
        template, mask = templates[number]
        # cv2.imshow('template', template)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)

        res = cv2.matchTemplate(img_cropped_gray, template, cv2.TM_CCOEFF_NORMED, mask=mask)
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
    # cv2.imshow('match', img_cropped_gray)
    # cv2.waitKey(0)

    return ult

def read_ults(img, stat_rects, ratio, templates):
    ults = {}
    for key in stat_rects:
        ults[key] = read_ult(img, stat_rects[key], ratio, templates)

    return ults

def read_match_ults():
    templates = read_templates(RATIO)
    stat_rects = read_stat_rects()

    data = {}
    data['ult'] = {}
    length = 732

    for key in stat_rects:
        data['ult'][key] = []

    for i in range(length):
        index = i*30
        # NOTE: Read image takes long time
        img = cv2.imread('img/overwatch_1_1_'+str(index)+'.jpg', cv2.IMREAD_COLOR)
        ults = read_ults(img, stat_rects, RATIO, templates)

        for key in ults:
            data['ult'][key].append(ults[key])
        # print('{:.0f}%'.format(100*i/(length-1)))

    return data

# data = read_match_ults()
#
# with open('data.json', 'w') as outfile:
#     json.dump(data, outfile)

with open('data_ults.json') as json_file:
    data = json.load(json_file)

end = 7
for i in range(1,end):
    plt.subplot(end,1,i)
    plt.plot(data['ult'][str(i)])
plt.show()

# NOTE: Rez and death color change
