import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

HEALTH_NUMBER_INDEX = [
    840,
    11490,
    4200,
    3810,
    20100,
    4170,
    17700,
    3690,
    3780,
    2400
]

HEALTH_NUMBER_LEFT = 204
HEALTH_NUMBER_TOP = 605
HEALTH_NUMBER_WIDTH = 8
HEALTH_NUMBER_HEIGHT = 20
HEALTH_NUMBER_TILT = 5

def number_mask_poly(x, y, w, h):
    poly = np.array([HEALTH_NUMBER_LEFT, HEALTH_NUMBER_TOP,
                     HEALTH_NUMBER_LEFT+(HEALTH_NUMBER_WIDTH-w), HEALTH_NUMBER_TOP,
                     HEALTH_NUMBER_LEFT+(HEALTH_NUMBER_WIDTH-w)-HEALTH_NUMBER_TILT, HEALTH_NUMBER_TOP+(HEALTH_NUMBER_HEIGHT-h),
                     HEALTH_NUMBER_LEFT-HEALTH_NUMBER_TILT, HEALTH_NUMBER_TOP+(HEALTH_NUMBER_HEIGHT-h)],
                     dtype=np.float32).reshape((-1,2))

    return poly-np.tile(np.array([x, y], dtype=np.float32),(4,1))

HEALTH_NUMBER_MASK_POLY = [
    number_mask_poly(-1, -1, 0, 0),
    number_mask_poly(3, -2, 4, 0),
    number_mask_poly(3, -1, 0, 0),
    number_mask_poly(3, -11, 0, 0),
    number_mask_poly(10, -1, 0, 0),
    number_mask_poly(4, 0, 0, 0),
    number_mask_poly(10, 0, 0, 0),
    number_mask_poly(4, 0, 1, 0),
    number_mask_poly(3, 0, 0, 0),
    number_mask_poly(3, 0, 0, 0),
]

HEALTH_RECT = (175, 595, 42, 55)

RATIO = 4
ALPHA = 1.5 # Scale
BETA = -127*ALPHA+50 # Offset
MATCH_THRESHOLD = 0.9
HEALTH_NUMBER_TOO_CLOSE_THRESHOLD = 7*RATIO

def crop(img, rect):
    img_crop = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    return img_crop

def save_templates():
    for i in range(len(HEALTH_NUMBER_INDEX)):
        img = cv2.imread('img/overwatch_1_1_'+str(HEALTH_NUMBER_INDEX[i])+'.jpg', cv2.IMREAD_COLOR)
        rect = cv2.boundingRect(HEALTH_NUMBER_MASK_POLY[i])
        # img = cv2.polylines(img, [HEALTH_NUMBER_MASK_POLY[i].astype(np.int32)], True, (255,255,255), thickness=1)
        img = crop(img, rect)
        cv2.imwrite('template/health_number_'+str(i)+'.jpg', img)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

def read_templates(ratio):
    templates = {}
    for i in range(10):
        rect = cv2.boundingRect(HEALTH_NUMBER_MASK_POLY[i])
        mask_poly_offset = HEALTH_NUMBER_MASK_POLY[i]-np.tile(rect[0:2],(4,1))

        template = cv2.imread('template/health_number_'+str(i)+'.jpg', cv2.IMREAD_COLOR)

        mask = np.zeros(template.shape[0:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, mask_poly_offset.astype(np.int32), (255,255,255))
        mask = cv2.resize(mask, None, fx=ratio, fy=ratio)

        template = preprocess_image(template, ratio)
        # template = cv2.resize(template, None, fx=ratio, fy=ratio)

        template[mask == 0] = 0

        cv2.imshow('template', template)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        templates[i] = (template, mask)

    return templates

def read_health(img, health_rect, ratio, templates):
    img_cropped = crop(img, health_rect)
    img_cropped = preprocess_image(img_cropped, ratio)
    # img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    # img_cropped = cv2.resize(img_cropped, None, fx=ratio, fy=ratio)
    # img_cropped = cv2.convertScaleAbs(img_cropped, alpha=ALPHA, beta=BETA)

    match = []
    for number in range(10):
        template, mask = templates[number]
        res = cv2.matchTemplate(img_cropped, template, cv2.TM_CCOEFF_NORMED, mask=mask)
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
                if (l[0]-l_simple[0])**2+(l[1]-l_simple[1])**2 < HEALTH_NUMBER_TOO_CLOSE_THRESHOLD **2:
                    tooClose = True
                    break
            if not tooClose:
                locations_simple.append(l)
                match.append((number, l, res[l[1],l[0]])) # Consider as two numbers

    for i in range(len(match)-1,-1,-1):
        # Remove match with inf res
        if match[i][2] == float('inf'):
            match.pop(i)

    if len(match) > 3:
        # Only save three matches with highest res
        match.sort(reverse=True, key=lambda l:l[2])
        match = match[0:3]

    # Sort according to x so that value is correct
    match.sort(key=lambda l:l[1][0])

    # Calculate ult value
    if len(match) == 3:
        health = match[0][0]*100+match[1][0]*10+match[2][0]
    elif len(match) == 2:
        health = match[0][0]*10+match[1][0]
    elif len(match) == 1:
        health = match[0][0]
    else:
        health = -1

    print(match)
    cv2.imshow('img', img_cropped)
    cv2.waitKey(0)

    return health

def remove_spikes(ult, window_size):
    i = 0
    while(i+window_size < len(ult)):
        delta_ult = ult[i+1:i+window_size]-ult[i:i+window_size-1]
        change_total = np.sum(np.absolute(delta_ult))
        change_net = np.absolute(np.sum(delta_ult))
        if change_net < change_total*0.2:
            ult[i+1:i+window_size-1] = (ult[i]+ult[i+window_size-1])/2
        i += 1

def test_read_health():
    save_templates()
    templates = read_templates(RATIO)

    data = []
    start = 350
    end = 400
    t = np.arange(start,end)
    for i in range(start, end):
        t0 = time.time()
        img = cv2.imread('img/overwatch_1_1_'+str(i*30)+'.jpg', cv2.IMREAD_COLOR)
        t1 = time.time()
        health = read_health(img, HEALTH_RECT, RATIO, templates)
        data.append(health)
        t2 = time.time()
        # print(t1-t0, t2-t1)
        # print('index', i*30, 'health', health)
        # cv2.imshow('src', img)
        # cv2.waitKey(0)

    data = np.array(data)
    data_adjusted = data.copy()
    remove_spikes(data_adjusted, 5)
    plt.plot(t, data)
    plt.plot(t, data_adjusted)
    plt.show()

def test_hls():
    for i in range(620, 720):
        img = cv2.imread('img/overwatch_1_1_'+str(i*30)+'.jpg', cv2.IMREAD_COLOR)
        img = crop(img, HEALTH_RECT)
        img = cv2.resize(img, None, fx=RATIO, fy=RATIO)

        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_l = img_hls[:,:,1] # Only lightness channel
        img_s = img_hls[:,:,2] # Only lightness channel

        # img_l = cv2.convertScaleAbs(img_l, alpha=ALPHA, beta=BETA)
        # img_gray = cv2.convertScaleAbs(img_gray, alpha=ALPHA, beta=BETA)

        cv2.imshow('lightness', img_l)
        cv2.imshow('saturation', img_s)
        cv2.imshow('gray', img_gray)
        cv2.waitKey(0)

def preprocess_image(img, ratio):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    img = cv2.convertScaleAbs(img, alpha=ALPHA, beta=BETA)
    img = cv2.medianBlur(img, 3)
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,6)

    return img

def test_preprocess():
    for i in range(40, 732):
        # print(i)
        img = cv2.imread('img/overwatch_1_1_'+str(i*30)+'.jpg', cv2.IMREAD_COLOR)
        img = crop(img, HEALTH_RECT)
        img = preprocess_image(img, RATIO)

        cv2.imshow('gray', img)
        cv2.waitKey(0)

def test_edge():
    num_width = 15
    num_height = 8

    for j in range(0,int(732/num_width/num_height)):
        imgs = []
        for i in range(j*num_width*num_height, j*num_width*num_height+num_width*num_height):
            img = cv2.imread('img/overwatch_1_1_'+str(i*30)+'.jpg', cv2.IMREAD_COLOR)
            img = crop(img, HEALTH_RECT)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, None, fx=2, fy=2)
            img = cv2.convertScaleAbs(img, alpha=ALPHA, beta=BETA)
            img = cv2.GaussianBlur(img, (5, 5), 5)

            img = cv2.Canny(img,100,200)
            # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)), iterations=1)
            img = cv2.morphologyEx(img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)

            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # img = np.zeros((img.shape[0],img.shape[1],3))
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100: continue
                img = cv2.polylines(img, [contour], True, (255,0,0))
            img = cv2.putText(img, str(i*30), (0,img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
            imgs.append(img)

            print(len(contours), i*30)

            # cv2.imshow('gray', img)
            # cv2.imshow('edges', edges)
            # cv2.waitKey(0)

        imgs_row = []
        for i in range(num_height):
            imgs_row.append(cv2.hconcat(imgs[i*num_width:i*num_width+num_width], num_width))

        img_final = cv2.vconcat(imgs_row, num_height)
        cv2.imshow('edges', img_final)
        cv2.waitKey(0)

# test_hls()
# test_read_health()
# test_preprocess()
# read_templates(RATIO)
test_edge()
