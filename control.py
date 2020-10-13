import cv2
import numpy as np

from utils import crop, match_color, read_batch

STATUS_RECT = (624,52,32,80)
LOCKED_RECT = (626,80,28,28)
UNLOCKED_RECT = (625,58,30,30)
STATUS_THRESHOLD = 0.5

def save_control_templates():
    img = cv2.imread('img/nepal/nepal_600.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_locked.jpg', crop(img, LOCKED_RECT))

    img = cv2.imread('img/nepal/nepal_2130.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_c0.jpg', crop(img, UNLOCKED_RECT))
    img = cv2.imread('img/nepal/nepal_2760.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_c1.jpg', crop(img, UNLOCKED_RECT))
    img = cv2.imread('img/nepal/nepal_6780.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_c2.jpg', crop(img, UNLOCKED_RECT))

    img = cv2.imread('img/nepal/nepal_11910.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_b0.jpg', crop(img, UNLOCKED_RECT))
    img = cv2.imread('img/nepal/nepal_12180.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_b1.jpg', crop(img, UNLOCKED_RECT))
    img = cv2.imread('img/nepal/nepal_13650.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_b2.jpg', crop(img, UNLOCKED_RECT))

    img = cv2.imread('img/nepal/nepal_20670.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_a0.jpg', crop(img, UNLOCKED_RECT))
    img = cv2.imread('img/control_a1/control_a1_240.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_a1.jpg', crop(img, UNLOCKED_RECT))
    img = cv2.imread('img/nepal/nepal_20880.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/control_a2.jpg', crop(img, UNLOCKED_RECT))

def read_control_tempaltes():
    templates = {}

    img_locked = cv2.imread('template/control_locked.jpg', cv2.IMREAD_COLOR)
    templates['locked'] = (img_locked, None)

    center = (int(UNLOCKED_RECT[2]/2-1),int(UNLOCKED_RECT[2]/2-1))
    mask_unlocked = np.zeros((UNLOCKED_RECT[3],UNLOCKED_RECT[2]), dtype=np.uint8)
    mask_unlocked = cv2.circle(mask_unlocked, center, 14, 255, thickness=-1)

    img_c0 = cv2.imread('template/control_c0.jpg', cv2.IMREAD_COLOR)
    templates['c0'] = (img_c0, mask_unlocked)
    img_c1 = cv2.imread('template/control_c1.jpg', cv2.IMREAD_COLOR)
    templates['c1'] = (img_c1, mask_unlocked)
    img_c2 = cv2.imread('template/control_c2.jpg', cv2.IMREAD_COLOR)
    templates['c2'] = (img_c2, mask_unlocked)

    img_b0 = cv2.imread('template/control_b0.jpg', cv2.IMREAD_COLOR)
    templates['b0'] = (img_b0, mask_unlocked)
    img_b1 = cv2.imread('template/control_b1.jpg', cv2.IMREAD_COLOR)
    templates['b1'] = (img_b1, mask_unlocked)
    img_b2 = cv2.imread('template/control_b2.jpg', cv2.IMREAD_COLOR)
    templates['b2'] = (img_b2, mask_unlocked)

    img_a0 = cv2.imread('template/control_a0.jpg', cv2.IMREAD_COLOR)
    templates['a0'] = (img_a0, mask_unlocked)
    img_a1 = cv2.imread('template/control_a1.jpg', cv2.IMREAD_COLOR)
    templates['a1'] = (img_a1, mask_unlocked)
    img_a2 = cv2.imread('template/control_a2.jpg', cv2.IMREAD_COLOR)
    templates['a2'] = (img_a2, mask_unlocked)
    # imt_tmp = img_c0.copy()
    # imt_tmp[mask_unlocked == 0] = (255,255,255)
    # cv2.imshow('Mask', mask_unlocked)
    # cv2.waitKey(0)

    return templates

def read_point_status(img, templates):
    img = crop(img, STATUS_RECT)
    scores = []

    template, mask = templates['locked']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.unravel_index(np.argmax(res), res.shape)
    scores.append((-1, None, loc, res[loc]))

    for map in ['a', 'b', 'c']:
        for state in [0,1,2]:
            template, mask = templates[map+str(state)]
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            loc = np.unravel_index(np.argmax(res), res.shape)
            scores.append((state, map, loc, res[loc]))

    scores.sort(reverse=True, key=lambda m:m[3])
    score = scores[0]

    if score[3] > STATUS_THRESHOLD:
        return score[0], score[1], (score[2][1],score[2][0])
    else:
        return None, None, None

save_control_templates()
# read_control_tempaltes()

templates = read_control_tempaltes()
def process_status(img):
    status, map, loc = read_point_status(img, templates)

    img = crop(img, STATUS_RECT)
    if status is None:
        return 'NA', img
    else:
        if map is None:
            return '{:d}'.format(status), img
        else:
            return '{}{:d}'.format(map, status), img

read_batch(process_status, start=4)
