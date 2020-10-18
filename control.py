import cv2
import numpy as np

from utils import crop, match_color, read_batch, val_to_string

STATUS_RECT = (624,52,32,80)
LOCKED_RECT = (626,80,28,28)
UNLOCKED_RECT = (625,58,30,30)
STATUS_THRESHOLD = 0.5

RATIO = 2.0
PERCENT_SYMBOL_1_RECT = (585,64,13,17)
PERCENT_SYMBOL_2_RECT = (690,64,13,17)
PERCENT_0_1_RECT = (577,64,8,17)
PERCENT_0_2_RECT = (682,64,8,17)
PERCENT_TILT_X = 4
PERCENT_TILT_Y = 8
PERCENT_0_MASK = np.array([
    [PERCENT_TILT_X, 0],
    [PERCENT_0_1_RECT[2]-1, 0],
    [PERCENT_0_1_RECT[2]-1, PERCENT_0_1_RECT[3]-1],
    [0, PERCENT_0_1_RECT[3]-1],
    [0, PERCENT_TILT_Y],
], dtype=np.int32)
PERCENT_1_1_RECT = (578,64,6,17)
PERCENT_1_2_RECT = (683,64,6,17)
PERCENT_RECTS = {
    0: {1: PERCENT_0_1_RECT, 2: PERCENT_0_2_RECT},
    1: {1: PERCENT_1_1_RECT, 2: PERCENT_1_2_RECT},
    2: {1: PERCENT_0_1_RECT, 2: PERCENT_0_2_RECT},
    3: {1: PERCENT_0_1_RECT, 2: PERCENT_0_2_RECT},
    4: {1: PERCENT_0_1_RECT, 2: PERCENT_0_2_RECT},
    5: {1: PERCENT_0_1_RECT, 2: PERCENT_0_2_RECT},
    6: {1: PERCENT_0_1_RECT, 2: PERCENT_0_2_RECT},
    7: {1: PERCENT_0_1_RECT, 2: PERCENT_0_2_RECT},
    8: {1: PERCENT_0_1_RECT, 2: PERCENT_0_2_RECT},
    9: {1: PERCENT_0_1_RECT, 2: PERCENT_0_2_RECT},
    'symbol': {1: PERCENT_SYMBOL_1_RECT, 2: PERCENT_SYMBOL_2_RECT}
}
PERCENT_THRESHOLD = 0.5

FULL_PROGRESS_RECT = (566,58,144,30)
T1_PROGRESS_RECT = (566,64,36,17)
T2_PROGRESS_RECT = (671,64,36,17)

def save_templates():
    img = cv2.imread('img/nepal/nepal_600.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_locked.jpg', crop(img, LOCKED_RECT))

    img = cv2.imread('img/nepal/nepal_2130.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_c_0.jpg', crop(img, UNLOCKED_RECT))
    img = cv2.imread('img/nepal/nepal_2760.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_c_1.jpg', crop(img, UNLOCKED_RECT))
    img = cv2.imread('img/nepal/nepal_6780.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_c_2.jpg', crop(img, UNLOCKED_RECT))

    img = cv2.imread('img/nepal/nepal_11910.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_b_0.jpg', crop(img, UNLOCKED_RECT))
    img = cv2.imread('img/nepal/nepal_12180.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_b_1.jpg', crop(img, UNLOCKED_RECT))
    img = cv2.imread('img/nepal/nepal_13650.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_b_2.jpg', crop(img, UNLOCKED_RECT))

    img = cv2.imread('img/nepal/nepal_20670.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_a_0.jpg', crop(img, UNLOCKED_RECT))
    img = cv2.imread('img/control_a1/control_a1_240.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_a_1.jpg', crop(img, UNLOCKED_RECT))
    img = cv2.imread('img/nepal/nepal_20880.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_a_2.jpg', crop(img, UNLOCKED_RECT))

    img = cv2.imread('img/nepal/nepal_2760.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_symbol_1.jpg', crop(img, PERCENT_RECTS['symbol'][1]))
    cv2.imwrite('template/control_0_1.jpg', crop(img, PERCENT_RECTS[0][1]))
    img = cv2.imread('img/nepal/nepal_2790.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_1_1.jpg', crop(img, PERCENT_RECTS[1][1]))
    img = cv2.imread('img/nepal/nepal_2820.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_2_1.jpg', crop(img, PERCENT_RECTS[2][1]))
    img = cv2.imread('img/nepal/nepal_2850.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_3_1.jpg', crop(img, PERCENT_RECTS[3][1]))
    img = cv2.imread('img/nepal/nepal_2880.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_4_1.jpg', crop(img, PERCENT_RECTS[4][1]))
    img = cv2.imread('img/nepal/nepal_2940.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_5_1.jpg', crop(img, PERCENT_RECTS[5][1]))
    img = cv2.imread('img/nepal/nepal_2970.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_6_1.jpg', crop(img, PERCENT_RECTS[6][1]))
    img = cv2.imread('img/nepal/nepal_3000.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_7_1.jpg', crop(img, PERCENT_RECTS[7][1]))
    img = cv2.imread('img/nepal/nepal_3030.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_8_1.jpg', crop(img, PERCENT_RECTS[8][1]))
    img = cv2.imread('img/nepal/nepal_3060.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_9_1.jpg', crop(img, PERCENT_RECTS[9][1]))

    img = cv2.imread('img/nepal/nepal_6270.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_symbol_2.jpg', crop(img, PERCENT_RECTS['symbol'][2]))
    cv2.imwrite('template/control_0_2.jpg', crop(img, PERCENT_RECTS[0][2]))
    img = cv2.imread('img/nepal/nepal_6300.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_1_2.jpg', crop(img, PERCENT_RECTS[1][2]))
    img = cv2.imread('img/nepal/nepal_6330.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_2_2.jpg', crop(img, PERCENT_RECTS[2][2]))
    img = cv2.imread('img/nepal/nepal_6390.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_3_2.jpg', crop(img, PERCENT_RECTS[3][2]))
    img = cv2.imread('img/nepal/nepal_6420.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_4_2.jpg', crop(img, PERCENT_RECTS[4][2]))
    img = cv2.imread('img/nepal/nepal_6450.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_5_2.jpg', crop(img, PERCENT_RECTS[5][2]))
    img = cv2.imread('img/nepal/nepal_6480.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_6_2.jpg', crop(img, PERCENT_RECTS[6][2]))
    img = cv2.imread('img/nepal/nepal_6510.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_7_2.jpg', crop(img, PERCENT_RECTS[7][2]))
    img = cv2.imread('img/nepal/nepal_6540.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_8_2.jpg', crop(img, PERCENT_RECTS[8][2]))
    img = cv2.imread('img/nepal/nepal_6600.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_9_2.jpg', crop(img, PERCENT_RECTS[9][2]))

    # cv2.imshow('img', crop(img, PERCENT_RECTS[1][2]))
    # cv2.waitKey(0)

def read_tempaltes():
    templates = {}

    img_locked = cv2.imread('template/control_locked.jpg', cv2.IMREAD_COLOR)
    templates['locked'] = (img_locked, None)

    center = (int(UNLOCKED_RECT[2]/2-1),int(UNLOCKED_RECT[2]/2-1))
    mask_unlocked = np.zeros((UNLOCKED_RECT[3],UNLOCKED_RECT[2]), dtype=np.uint8)
    mask_unlocked = cv2.circle(mask_unlocked, center, 14, 255, thickness=-1)

    mask_percent_0 = np.zeros((PERCENT_0_1_RECT[3],PERCENT_0_1_RECT[2]), dtype=np.uint8)
    mask_percent_0 = cv2.fillConvexPoly(mask_percent_0, PERCENT_0_MASK, 255)
    mask_percent_0 = cv2.resize(mask_percent_0, None, fx=RATIO, fy=RATIO)

    mask_percent_1 = np.zeros((PERCENT_1_1_RECT[3],PERCENT_1_1_RECT[2]), dtype=np.uint8)
    mask_percent_1 = cv2.resize(mask_percent_1, None, fx=RATIO, fy=RATIO)
    mask_percent_1[:,:] = 255

    # Status
    for map in ['a', 'b', 'c']:
        templates[map] = {}
        for t in [0,1,2]:
            img_t = cv2.imread('template/control_'+map+'_'+str(t)+'.jpg', cv2.IMREAD_COLOR)
            templates[map][t] = (img_t, mask_unlocked)

    # Number
    for num in range(10):
        templates[num] = {}
        for t in range(1,3):
            img_num = cv2.imread('template/control_'+str(num)+'_'+str(t)+'.jpg', cv2.IMREAD_COLOR)
            img_num = cv2.resize(img_num, None, fx=RATIO, fy=RATIO)

            if num == 1:
                mask = mask_percent_1
            else:
                mask = mask_percent_0

            templates[num][t] = (img_num, mask)


    # Symbol
    templates['symbol'] = {}
    for t in range(1,3):
        img_sym = cv2.imread('template/control_symbol_'+str(t)+'.jpg', cv2.IMREAD_COLOR)
        templates['symbol'][t] = (img_sym, None)

    # temp, mask = templates[7][2]
    # temp[mask == 0] = (0,0,0)
    # cv2.imshow('temp', temp)
    # cv2.waitKey(0)

    return templates

def read_status(img, templates):
    img = crop(img, STATUS_RECT)
    scores = []

    template, mask = templates['locked']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.unravel_index(np.argmax(res), res.shape)
    scores.append((-1, None, loc, res[loc]))

    for map in ['a', 'b', 'c']:
        for status in [0,1,2]:
            template, mask = templates[map][status]
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            loc = np.unravel_index(np.argmax(res), res.shape)
            scores.append((status, map, loc, res[loc]))

    scores.sort(reverse=True, key=lambda m:m[3])
    score = scores[0]

    if score[3] > STATUS_THRESHOLD:
        return score[0], score[1], (score[2][1],score[2][0])
    else:
        return None, None, None

# None, None, None: if no progress UI detected
# None -1, None: if point is locked
# map, 0, 0: if point is unlocked but not captured.
# map, 1, percent: if point is captured by team 1.
# map, 2, percent: if point is captured by team 2.
def read_progress(src, templates):
    status, map, loc = read_status(src, templates)

    img_full_progress = crop(src, FULL_PROGRESS_RECT)
    if status is None: return None, None, None
    if status == -1: return None, -1, None
    if status == 0: return map, 0, 0

    if status == 1:
        t_progress_rect = list(T1_PROGRESS_RECT)
    else:
        t_progress_rect = list(T2_PROGRESS_RECT)
    # Only two possible values since only overtime or not
    # related to y of STATUS_RECT
    # print(loc[1])
    if loc[1] > 10:
        dy = 43-(UNLOCKED_RECT[1]-STATUS_RECT[1])
    else:
        dy = 6-(UNLOCKED_RECT[1]-STATUS_RECT[1])
    t_progress_rect[1] += dy
    full_progress_rect = list(FULL_PROGRESS_RECT)
    full_progress_rect[1] += dy

    img = crop(src, t_progress_rect)
    # Find symbol location
    template, mask = templates['symbol'][status]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.unravel_index(np.argmax(res), res.shape)

    # Only two possible values since max two digits
    # related to x of t_progress_rect
    # print(loc[1])
    if loc[1] > 19:
        dx = 22
    else:
        dx = 19

    padx = 1

    digit_1_scores = []
    for num in range(10):
        template, mask = templates[num][status]
        w_digit_1 = PERCENT_RECTS[num][status][2]
        digit_1_rect = (t_progress_rect[0]+dx-w_digit_1-padx, t_progress_rect[1], w_digit_1+padx*2, t_progress_rect[3])
        img_digit_1 = crop(src, digit_1_rect)
        img_digit_1_scaled = cv2.resize(img_digit_1, None, fx=RATIO, fy=RATIO)
        res = cv2.matchTemplate(img_digit_1_scaled, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        digit_1_scores.append((num, np.max(res)))
    digit_1_scores.sort(reverse=True, key=lambda s:s[1])
    digit_1 = digit_1_scores[0][0]
    if digit_1_scores[0][1] < PERCENT_THRESHOLD:
        return map, status, None

    digit_2_scores = []
    for num in range(10):
        template, mask = templates[num][status]
        w_digit_2 = PERCENT_RECTS[num][status][2]
        w_digit_1 = PERCENT_RECTS[digit_1][status][2]
        digit_2_rect = (t_progress_rect[0]+dx-w_digit_2-w_digit_1-padx+1, t_progress_rect[1], w_digit_2+padx*2, t_progress_rect[3])
        img_digit_2 = crop(src, digit_2_rect)
        img_digit_2_scaled = cv2.resize(img_digit_2, None, fx=RATIO, fy=RATIO)
        res = cv2.matchTemplate(img_digit_2_scaled, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        digit_2_scores.append((num, np.max(res)))
    digit_2_scores.sort(reverse=True, key=lambda s:s[1])
    digit_2 = digit_2_scores[0][0]
    if digit_2_scores[0][1] < PERCENT_THRESHOLD: digit_2 = 0

    # print(digit_1_scores)
    # print(digit_2_scores)
    percent = digit_2*10+digit_1

    return map, status, percent

templates = read_tempaltes()

def process_status(img):
    status, map, loc = read_status(img, templates)

    img = crop(img, STATUS_RECT)
    return '{}{}'.format(
        val_to_string(map),
        val_to_string(status)
    ), img

def process_progress(img):
    img_full_progress = crop(img, FULL_PROGRESS_RECT)

    map, status, percent = read_progress(img, templates)

    return '{} {} {}'.format(
        val_to_string(map),
        val_to_string(status),
        val_to_string(percent)
    ), img_full_progress

# save_templates()
# tempaltes = read_tempaltes()

# read_batch(process_status, start=4)
read_batch(process_progress, start=0, num_height=16)
