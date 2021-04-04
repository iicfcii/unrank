import cv2
import numpy as np
import matplotlib.pyplot as plt

import utils
import assault
import escort

LOCKED_RECT = (605,82,25,25)
LOCKED_MASK = np.array([
    [int(LOCKED_RECT[2]/2), 0],
    [LOCKED_RECT[2]-1, int(LOCKED_RECT[3]/2)],
    [int(LOCKED_RECT[2]/2), LOCKED_RECT[3]-1],
    [0,int(LOCKED_RECT[3]/2)],
], dtype=np.int32)

STATUS_RECT = (489,71,303,46)
STATUS_POINT_RECT = (594,71,46,46)
STATUS_POINT_B_RECT = (642,71,46,46)
STATUS_POINT_CAPTURED_RECT = (489,71,46,46)
STATUS_PAYLOAD_RECT = (538,71,escort.PAYLOAD_RECT_WIDTH,46)
STATUS_THRESHOLD = 0.6

def save_templates():
    img = cv2.imread('template_src/numbani_1530.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hybrid_locked.jpg', utils.crop(img, LOCKED_RECT))

    img = cv2.imread('template_src/numbani_14670.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hybrid_captured.jpg', utils.crop(img, LOCKED_RECT))

    img = cv2.imread('template_src/numbani_2070.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hybrid_point_1.jpg', utils.crop(img, LOCKED_RECT))

    img = cv2.imread('template_src/numbani_16650.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hybrid_point_2.jpg', utils.crop(img, LOCKED_RECT))

    # cv2.imshow('img', utils.crop(img, LOCKED_RECT))
    # cv2.waitKey(0)

def read_tempaltes():
    templates = {}

    img_locked = cv2.imread('template/hybrid_locked.jpg', cv2.IMREAD_COLOR)
    mask_locked = np.zeros((LOCKED_RECT[3],LOCKED_RECT[2]), dtype=np.uint8)
    mask_locked = cv2.fillConvexPoly(mask_locked, LOCKED_MASK, 255)
    templates['locked'] = (img_locked, mask_locked)

    img_captured = cv2.imread('template/hybrid_captured.jpg', cv2.IMREAD_COLOR)
    mask_captured = np.zeros((LOCKED_RECT[3],LOCKED_RECT[2]), dtype=np.uint8)
    mask_captured = cv2.fillConvexPoly(mask_captured, LOCKED_MASK, 255)
    templates['captured'] = (img_captured, mask_captured)

    templates['point'] = {}
    for team in [1,2]:
        img = cv2.imread('template/hybrid_point_'+str(team)+'.jpg', cv2.IMREAD_COLOR)
        mask = np.zeros((LOCKED_RECT[3],LOCKED_RECT[2]), dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, LOCKED_MASK, 255)
        templates['point'][team] = (img, mask)

    templates['payload'] = {}
    for team in [1,2]:
        img = cv2.imread('template/escort_payload_'+str(team)+'.jpg', cv2.IMREAD_COLOR)
        mask = np.zeros((escort.PAYLOAD_RECT[3],escort.PAYLOAD_RECT[2]), dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, escort.PAYLOAD_MASK, 255)
        templates['payload'][team] = (img, mask)

    # img, mask = templates['payload'][2]
    # img[mask == 0] = (0,0,0)
    # cv2.imshow('temp', img)
    # cv2.waitKey(0)

    return templates

def read_status(src, templates):
    img = utils.crop(src, STATUS_POINT_RECT)

    scores = []
    template, mask = templates['locked']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    loc = np.unravel_index(np.argmax(res), res.shape)
    scores.append(('point', -1, loc, res[loc]))

    template, mask = templates['captured']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    loc = np.unravel_index(np.argmax(res), res.shape)
    scores.append(('point', 0, loc, res[loc]))

    for team in [1,2]:
        template, mask = templates['point'][team]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        loc = np.unravel_index(np.argmax(res), res.shape)
        scores.append(('point', team, loc, res[loc]))

    img = utils.crop(src, STATUS_POINT_CAPTURED_RECT)
    template, mask = templates['captured']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    loc = np.unravel_index(np.argmax(res), res.shape)
    scores.append(('point', -2, loc, res[loc])) # Point captured and payload moving

    img = utils.crop(src, STATUS_PAYLOAD_RECT)
    for team in [1,2]:
        template, mask = templates['payload'][team]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        loc = np.unravel_index(np.argmax(res), res.shape)
        scores.append(('payload', team, loc, res[loc]))

    scores.sort(reverse=True, key=lambda m:m[3])
    score = scores[0]
    # print(scores)
    if score[0] == 'point' and score[1] > -2: # Payload not moving
        if score[3] > STATUS_THRESHOLD:
            if score[1] > 0:
                dy = 15 if score[2][0] > 12 else 11
                rect_point = (
                    assault.POINT_RECT[0],
                    assault.POINT_RECT[1]+(dy-LOCKED_RECT[1]+STATUS_POINT_RECT[1]),
                    assault.POINT_RECT[2],
                    assault.POINT_RECT[3]
                )
                rect_text = (
                    assault.TEXT_RECT[0],
                    assault.TEXT_RECT[1]+(dy-LOCKED_RECT[1]+STATUS_POINT_RECT[1]),
                    assault.TEXT_RECT[2],
                    assault.TEXT_RECT[3]
                )
                img = utils.crop(src, STATUS_POINT_RECT)
                img_point = utils.crop(img, rect_point)
                img_text = utils.crop(img, rect_text)
                if assault.read_capture(img_point, img_text,score[1],'A'):
                    # cv2.imshow('point', img_point)
                    # cv2.imshow('text', img_text)
                    # cv2.waitKey(0)
                    score = (score[0], score[1]+0.1, score[2], score[3])

            return score[1], (score[2][1],score[2][0]), -1, None # Defending team
        else:
            img = utils.crop(src, STATUS_POINT_B_RECT)
            # cv2.imshow('point b', img)
            # cv2.waitKey(0)

            scores = []
            template, mask = templates['locked']
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            loc = np.unravel_index(np.argmax(res), res.shape)
            scores.append((-1, loc, res[loc]))

            template, mask = templates['captured']
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            loc = np.unravel_index(np.argmax(res), res.shape)
            scores.append((0, loc, res[loc]))

            scores.sort(reverse=True, key=lambda m:m[2])
            score_alt = scores[0]
            # print(score_alt)
            if score_alt[2] > STATUS_THRESHOLD:
                return None, None, score_alt[0], (score_alt[1][1],score_alt[1][0])
            else:
                return None, None, None, None
    elif score[0] == 'point' and score[1] == -2: # Payload moving
        score_payload = [s for s in scores if s[0] == 'payload'][0] # First score with payload
        if score_payload[3] > escort.PAYLOAD_THRESHOLD:
            return 0, None, score_payload[1], (score_payload[2][1],score_payload[2][0]) # Attacking team
        else:
            return None, None, None, None
    else: # Must be a payload score
        if score[3] > STATUS_THRESHOLD:
            return 0, None, score[1], (score[2][1],score[2][0]) # Attacking team
        else:
            return None, None, None, None

def read_progress(src, templates):
    status_a, loc_a, status_b, loc_b = read_status(src, templates)
    img_progress = utils.crop(src, STATUS_RECT)

    if status_a is None or status_b is None: return None, None, None
    if status_a == -1: return -1, -1, -1
    if status_a == 0 and status_b == 0: return -1, 100, 100
    if status_a == 0 and status_b == -1: return -1, 100, -1

    if status_a > 0:
        team = np.floor(status_a)
        loc = loc_a

        # Overtime will offset in y direction
        # print(loc[1])
        dy = 15 if loc[1] > 12 else 11
        img= utils.crop(src, STATUS_POINT_RECT)
        center = (
            int(STATUS_POINT_RECT[2]/2),
            int(STATUS_POINT_RECT[3]/2+(dy-LOCKED_RECT[1]+STATUS_POINT_RECT[1]))
        )
        percent = assault.read_point(img, center, team)
        if percent is None: percent = 0

        return status_a, percent, -1
    else:
        # Get defending team
        team = 2 if status_b == 1 else 1
        loc = loc_b

        percent = escort.read_payload(loc[0])

        return team, 100, percent # Defending team

save_templates()
templates = read_tempaltes()

def process_status(src):
    status_a, loc_a, status_b, loc_b = read_status(src, templates)

    img = utils.crop(src, STATUS_RECT)

    return '{} {} '.format(
        utils.val_to_string(status_a),
        utils.val_to_string(status_b)
    ), img

def process_progress(img):
    status, progress_point, progress_payload = read_progress(img, templates)
    img_progress = utils.crop(img, STATUS_RECT)

    assault.mark_progress(
        img_progress,
        progress_point,
        STATUS_POINT_RECT[0]-STATUS_RECT[0]+STATUS_POINT_RECT[2]/2,
        STATUS_RECT[3]/2
    )
    escort.mark_progress(img_progress, progress_payload, STATUS_PAYLOAD_RECT[0]-STATUS_RECT[0], 2)

    return '{} {} {}'.format(
        utils.val_to_string(status),
        utils.val_to_string(progress_point),
        utils.val_to_string(progress_payload)
    ), img_progress

def save(start, end, code):
    obj = {
        'type':'hybrid',
        'status': [],
        'capturing': [],
        'progress': {
            'point': [],
            'payload': []
        },
    }

    for src, frame in utils.read_frames(start, end, code):
        status, progress_point, progress_payload = read_progress(src, templates)

        team = int(np.floor(status)) if status is not None else None
        capturing = int(10*(status-team)) if status is not None else None
        obj['status'].append(team)
        obj['capturing'].append(capturing)
        obj['progress']['point'].append(progress_point)
        obj['progress']['payload'].append(progress_payload)

    utils.save_data('obj', obj, start, end, code)

def refine(code):
    obj = utils.load_data('obj',0,None,code)

    obj['status'] = utils.remove_outlier(obj['status'],6,interp=False)
    obj['capturing'] = utils.remove_outlier(obj['capturing'],3,['none','number'])
    obj['capturing'] = utils.remove_outlier(obj['capturing'],1,['change'])

    obj['progress']['point'] = utils.remove_outlier(obj['progress']['point'],6,['none','number'])
    obj['progress']['point'] = utils.remove_outlier(obj['progress']['point'],3,['change'])
    assault.remove_capture(obj['capturing'], obj['progress']['point'])

    # Avoid interpolation between -1 and other value
    # Fill none bewteen number and -1 with number
    obj['progress']['payload'] = utils.remove_outlier(obj['progress']['payload'],6,['none','number'],min=0)
    obj['progress']['payload'] = utils.remove_outlier(obj['progress']['payload'],3,['change'],min=0)
    # Change -1 none 100 -1 to -1 -1 -1 -1
    obj['progress']['payload'] = utils.remove_outlier(obj['progress']['payload'],2,interp=False)

    utils.extend_none(obj['status'], [
        obj['status'],
        obj['capturing'],
        obj['progress']['point'],
        obj['progress']['payload']
    ], type='left')

    obj_src = utils.load_data('obj',0,None,code)
    plt.figure('status')
    plt.plot(obj['status'])
    utils.save_fig(utils.file_path('fig_status',0,len(obj['status'])-1,code,ext='png'))

    plt.figure('capturing')
    plt.plot(obj['capturing'])
    plt.plot(obj_src['capturing'], '.', markersize=1)
    utils.save_fig(utils.file_path('fig_capturing',0,len(obj['status'])-1,code,ext='png'))

    plt.figure('progress')
    plt.plot(obj['progress']['point'])
    plt.plot(obj['progress']['payload'])
    plt.plot(obj_src['progress']['point'], '.', markersize=1)
    plt.plot(obj_src['progress']['payload'], '.', markersize=1)
    utils.save_fig(utils.file_path('fig_progress',0,len(obj['status'])-1,code,ext='png'))
    # plt.show()

    utils.save_data('obj_r', obj, 0, None, code)
