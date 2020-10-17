import cv2
import numpy as np

from utils import crop, match_color, read_batch, val_to_string
import assult
import escort

LOCKED_RECT = (605,82,25,25)
LOCKED_MASK = np.array([
    [int(LOCKED_RECT[2]/2), 0],
    [LOCKED_RECT[2]-1, int(LOCKED_RECT[3]/2)],
    [int(LOCKED_RECT[2]/2), LOCKED_RECT[3]-1],
    [0,int(LOCKED_RECT[3]/2)],
], dtype=np.int32)

PAYLOAD_RECT = (544,76,escort.PAYLOAD_WIDTH,escort.PAYLOAD_HEIGHT)

STATUS_RECT = (489,71,303,46)
STATUS_POINT_RECT = (594,71,46,46)
STATUS_POINT_CAPTURED_RECT = (489,71,46,46)
STATUS_PAYLOAD_RECT = (538,71,escort.PAYLOAD_RECT_WIDTH,46)
STATUS_THRESHOLD = 0.6

def save_templates():
    img = cv2.imread('img/numbani/numbani_1530.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hybrid_locked.jpg', crop(img, LOCKED_RECT))

    img = cv2.imread('img/numbani/numbani_14670.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hybrid_captured.jpg', crop(img, LOCKED_RECT))

    img = cv2.imread('img/numbani/numbani_2070.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hybrid_point_1.jpg', crop(img, LOCKED_RECT))

    img = cv2.imread('img/numbani/numbani_16650.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hybrid_point_2.jpg', crop(img, LOCKED_RECT))

    img = cv2.imread('img/numbani/numbani_8070.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hybrid_payload_2.jpg', crop(img, PAYLOAD_RECT))

    img = cv2.imread('img/numbani/numbani_17850.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hybrid_payload_1.jpg', crop(img, PAYLOAD_RECT))

    # cv2.imshow('img', crop(img, PAYLOAD_RECT))
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
        img = cv2.imread('template/hybrid_payload_'+str(team)+'.jpg', cv2.IMREAD_COLOR)
        mask = np.zeros((PAYLOAD_RECT[3],PAYLOAD_RECT[2]), dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, escort.PAYLOAD_MASK, 255)
        templates['payload'][team] = (img, mask)

    # img, mask = templates['payload'][2]
    # img[mask == 0] = (0,0,0)
    # cv2.imshow('temp', img)
    # cv2.waitKey(0)

    return templates

def read_status(src, templates):
    img = crop(src, STATUS_POINT_RECT)

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

    img = crop(src, STATUS_POINT_CAPTURED_RECT)
    template, mask = templates['captured']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    loc = np.unravel_index(np.argmax(res), res.shape)
    scores.append(('point', -2, loc, res[loc])) # Point captured and payload moving

    img = crop(src, STATUS_PAYLOAD_RECT)
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
            return score[1], (score[2][1],score[2][0]), -1, None # Defending team
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
    img_progress = crop(src, STATUS_RECT)

    if status_a is None or status_b is None: return None, None, None
    if status_a == -1: return None, -1, -1
    if status_a == 0 and status_b < 1: return None, 100, -1

    if status_a > 0:
        team = status_a
        loc = loc_a

        # Overtime will offset in y direction
        # print(loc[1])
        dy = 15 if loc[1] > 12 else 11
        img= crop(src, STATUS_POINT_RECT)
        center = (
            int(STATUS_POINT_RECT[2]/2),
            int(STATUS_POINT_RECT[3]/2+(dy-LOCKED_RECT[1]+STATUS_POINT_RECT[1]))
        )
        percent = assult.read_point(img, center, team)
        if percent is None: percent = 0

        return team, percent, -1
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

    img = crop(src, STATUS_RECT)

    return '{} {} '.format(
        val_to_string(status_a),
        val_to_string(status_b)
    ), img

def process_progress(img):
    team, progress_point, progress_payload = read_progress(img, templates)
    img_progress = crop(img, STATUS_RECT)

    assult.mark_progress(
        img_progress,
        progress_point,
        STATUS_POINT_RECT[0]-STATUS_RECT[0]+STATUS_POINT_RECT[2]/2,
        STATUS_RECT[3]/2
    )
    escort.mark_progress(img_progress, progress_payload, STATUS_PAYLOAD_RECT[0]-STATUS_RECT[0], 2)

    return '{} {} {}'.format(
        val_to_string(team),
        val_to_string(progress_point),
        val_to_string(progress_payload)
    ), img_progress


# read_batch(process_status, start=5, map='numbani', length=685, num_width=6, num_height=12)
# read_batch(process_progress, start=0, map='numbani', length=685, num_width=6, num_height=12)
