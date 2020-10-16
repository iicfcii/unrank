import cv2
import numpy as np

from utils import crop, match_color, read_batch

STATUS_RECT = (514,71,254,42)
LOCKED_RECT = (627,80,25,25)
LOCKED_MASK = np.array([
    [int(LOCKED_RECT[2]/2), 0],
    [LOCKED_RECT[2]-1, int(LOCKED_RECT[3]/2)],
    [int(LOCKED_RECT[2]/2), LOCKED_RECT[3]-1],
    [0,int(LOCKED_RECT[3]/2)],
], dtype=np.int32)

PAYLOAD_RECT = (519,76,9,13)
PAYLOAD_TILT_Y = 4
PAYLOAD_MASK = np.array([
    [0, 0],
    [PAYLOAD_RECT[2]-1, 0],
    [PAYLOAD_RECT[2]-1, PAYLOAD_RECT[3]-PAYLOAD_TILT_Y],
    [int(PAYLOAD_RECT[2]/2),PAYLOAD_RECT[3]-1],
    [0,PAYLOAD_RECT[3]-PAYLOAD_TILT_Y],
], dtype=np.int32)
PAYLOAD_START_X = 5
PAYLOAD_END_X = 240

STATUS_THRESHOLD = 0.6

def save_templates():
    img = cv2.imread('img/rialto/rialto_10560.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/escort_locked.jpg', crop(img, LOCKED_RECT))

    img = cv2.imread('img/rialto/rialto_1530.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/escort_payload_2.jpg', crop(img, PAYLOAD_RECT))

    img = cv2.imread('img/rialto/rialto_11610.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/escort_payload_1.jpg', crop(img, PAYLOAD_RECT))

    # cv2.imshow('img', crop(img, PAYLOAD_RECT))
    # cv2.waitKey(0)

def read_tempaltes():
    templates = {}

    img_locked = cv2.imread('template/escort_locked.jpg', cv2.IMREAD_COLOR)
    mask_locked = np.zeros((LOCKED_RECT[3],LOCKED_RECT[2]), dtype=np.uint8)
    mask_locked = cv2.fillConvexPoly(mask_locked, LOCKED_MASK, 255)
    templates['locked'] = (img_locked, mask_locked)

    templates['payload'] = {}
    for team in [1,2]:
        img = cv2.imread('template/escort_payload_'+str(team)+'.jpg', cv2.IMREAD_COLOR)
        mask = np.zeros((PAYLOAD_RECT[3],PAYLOAD_RECT[2]), dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, PAYLOAD_MASK, 255)
        templates['payload'][team] = (img, mask)

    # locked, locked_mask = templates['locked']
    # locked[locked_mask == 0] = (0,0,0)
    # cv2.imshow('locked', locked)
    # payload, payload_mask = templates['payload'][1]
    # payload[payload_mask == 0] = (0,0,0)
    # cv2.imshow('payload', payload)
    # cv2.waitKey(0)

    return templates

def read_status(src, templates):
    img = crop(src, STATUS_RECT)

    scores = []
    template, mask = templates['locked']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    loc = np.unravel_index(np.argmax(res), res.shape)
    scores.append((-1, loc, res[loc]))

    for team in [1,2]:
        template, mask = templates['payload'][team]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        loc = np.unravel_index(np.argmax(res), res.shape)
        scores.append((team, loc, res[loc]))

    scores.sort(reverse=True, key=lambda m:m[2])
    score = scores[0]

    if score[2] > STATUS_THRESHOLD:
        return score[0], (score[1][1],score[1][0])
    else:
        return None, None

def read_progress(src, templates):
    status, loc = read_status(src, templates)

    if status is None: return None, None
    if status == -1: return -1, 0

    percent = round((loc[0]-PAYLOAD_START_X)/(PAYLOAD_END_X-PAYLOAD_START_X)*100)

    return status, percent

save_templates()
templates = read_tempaltes()

def process_status(src):
    status, loc = read_status(src, templates)

    img = crop(src, STATUS_RECT)

    if status is None: return 'NA'.format(status), img
    return '{:d}'.format(status), img

def process_progress(img):
    img_progress = crop(img, STATUS_RECT)
    status, percent = read_progress(img, templates)

    if status is None: return 'NA', img_progress

    x =  int(percent/100*(PAYLOAD_END_X-PAYLOAD_START_X)+PAYLOAD_START_X+4)
    y =  10
    img_progress = cv2.circle(img_progress, (x,y), 5, 0, thickness=-1)


    return '{:d} {:d}'.format(status, percent), img_progress

# read_batch(process_status, start=0, map='rialto', length=470, num_width=6, num_height=12)
read_batch(process_progress, start=0, map='rialto', length=470, num_width=6, num_height=12)
