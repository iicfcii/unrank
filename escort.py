import cv2
import numpy as np
import matplotlib.pyplot as plt

import utils

LOCKED_RECT = (627,80,25,25)
LOCKED_MASK = np.array([
    [int(LOCKED_RECT[2]/2), 0],
    [LOCKED_RECT[2]-1, int(LOCKED_RECT[3]/2)],
    [int(LOCKED_RECT[2]/2), LOCKED_RECT[3]-1],
    [0,int(LOCKED_RECT[3]/2)],
], dtype=np.int32)

PAYLOAD_START_X = 5
PAYLOAD_END_X = 240
PAYLOAD_WIDTH = 9
PAYLOAD_HEIGHT = 15
PAYLOAD_RECT_WIDTH = 254
PAYLOAD_RECT = (519,75,PAYLOAD_WIDTH,PAYLOAD_HEIGHT)
PAYLOAD_TILT_Y = 4
PAYLOAD_MASK = np.array([
    [0, 0],
    [PAYLOAD_RECT[2]-1, 0],
    [PAYLOAD_RECT[2]-1, PAYLOAD_RECT[3]-PAYLOAD_TILT_Y],
    [int(PAYLOAD_RECT[2]/2),PAYLOAD_RECT[3]-1],
    [0,PAYLOAD_RECT[3]-PAYLOAD_TILT_Y],
], dtype=np.int32)
PAYLOAD_THRESHOLD = 0.58

STATUS_RECT = (514,71,PAYLOAD_RECT_WIDTH,42)
STATUS_THRESHOLD = 0.8

def save_templates():
    img = cv2.imread('template_src/rialto_10560.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/escort_locked.jpg', utils.crop(img, LOCKED_RECT))

    img = cv2.imread('template_src/rialto_1530.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/escort_payload_2.jpg', utils.crop(img, PAYLOAD_RECT))

    img = cv2.imread('template_src/rialto_11610.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/escort_payload_1.jpg', utils.crop(img, PAYLOAD_RECT))

    # cv2.imshow('img', utils.crop(img, PAYLOAD_RECT))
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
    img = utils.crop(src, STATUS_RECT)

    scores = []
    template, mask = templates['locked']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    res[np.isnan(res)] = 0
    loc = np.unravel_index(np.argmax(res), res.shape)
    scores.append((-1, loc, res[loc]))

    for team in [1,2]:
        template, mask = templates['payload'][team]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        res[np.isnan(res)] = 0
        loc = np.unravel_index(np.argmax(res), res.shape)
        scores.append((team, loc, res[loc]))

    scores.sort(reverse=True, key=lambda m:m[2])
    score = scores[0]
    # print(scores)
    threshold = STATUS_THRESHOLD if score[0] == -1 else PAYLOAD_THRESHOLD
    if score[2] > threshold:
        return score[0], (score[1][1],score[1][0]) # Attacking team
    else:
        return None, None

def read_payload(x):
    return round((x-PAYLOAD_START_X)/(PAYLOAD_END_X-PAYLOAD_START_X)*100)

def read_progress(src, templates):
    status, loc = read_status(src, templates)

    if status is None: return None, None
    if status == -1: return -1, -1

    team = 2 if status == 1 else 1
    percent = read_payload(loc[0])

    return team, percent  # Defending team

save_templates()
templates = read_tempaltes()

def process_status(src):
    status, loc = read_status(src, templates)

    img = utils.crop(src, STATUS_RECT)

    return '{}'.format(utils.val_to_string(status)), img

def mark_progress(img, progress, dx, dy):
    if progress is not None and progress > -1:
        dx = PAYLOAD_START_X+PAYLOAD_WIDTH/2+dx
        x =  int(progress/100*(PAYLOAD_END_X-PAYLOAD_START_X)+dx)
        y =  18+dy
        img = cv2.circle(img, (x,y), 3, 0, thickness=-1)

    return img

def process_progress(img):
    status, progress = read_progress(img, templates)
    img_progress = utils.crop(img, STATUS_RECT)

    mark_progress(img_progress, progress, 0, 0)

    return '{} {}'.format(
        utils.val_to_string(status),
        utils.val_to_string(progress)
    ), img_progress

def save(start, end, code):
    obj = {
        'type':'escort',
        'status': [],
        'progress': [],
    }

    for src, frame in utils.read_frames(start, end, code):
        status, progress = read_progress(src, templates)

        obj['status'].append(status)
        obj['progress'].append(progress)

    utils.save_data('obj', obj, start, end, code)

def refine(code):
    obj = utils.load_data('obj',0,None,code)

    # Fixed case: none num none first
    # This could happen when some templates match black screen
    obj['status'] = utils.remove_outlier(obj['status'],size=1,types=['number'],interp=False)
    obj['status'] = utils.remove_outlier(obj['status'],size=3,interp=False)
    # Avoid interpolation between -1 and other value
    obj['progress'] = utils.remove_outlier(obj['progress'],size=2,min=0)
    # Fill none bewteen number and -1 with number
    obj['progress'] = utils.remove_outlier(obj['progress'],size=2,interp=False)

    utils.extend_none(obj['status'], [
        obj['status'],
        obj['progress']
    ], type='left')

    obj_src = utils.load_data('obj',0,None,code)
    plt.figure('status')
    plt.plot(obj['status'])
    utils.save_fig(utils.file_path('fig_status',0,len(obj['status'])-1,code,ext='png'))

    plt.figure('progress')
    plt.plot(obj['progress'])
    plt.plot(obj_src['progress'], '.', markersize=1)
    utils.save_fig(utils.file_path('fig_progress',0,len(obj['status'])-1,code,ext='png'))
    # plt.show()

    utils.save_data('obj_r', obj, 0, None, code)
