import cv2
import numpy as np
import matplotlib.pyplot as plt

import utils

PROGRESS_RECT = (592,71,96,46)
PROGRESS_RECT_1 = (592,71,46,46)
PROGRESS_RECT_2 = (642,71,46,46)

ICON_RECT_1 = (601,81,26,26)
ICON_RECT_2 = (651,81,26,26)
ICON_MASK = np.array([[ICON_RECT_1[2]/2, 2],
                      [ICON_RECT_1[2]-2, ICON_RECT_1[3]/2],
                      [ICON_RECT_1[2]/2, ICON_RECT_1[3]-2],
                      [2, ICON_RECT_1[3]/2]], dtype=np.int32)
ICON_THRESHOLD = 0.6 # NOTE: A,B icon change size a bit when point is contesting

TEAM1_COLOR = np.array((85, 150, 255)) # HSV, RGB 67, 212, 255
TEAM1_COLOR_RANGE = np.array((20, 120, 40))
TEAM1_COLOR_LB = TEAM1_COLOR-TEAM1_COLOR_RANGE
TEAM1_COLOR_UB = TEAM1_COLOR+TEAM1_COLOR_RANGE

TEAM2_COLOR = np.array((170, 180, 230)) # HSV, RGB 240, 14, 54
TEAM2_COLOR_RANGE = np.array((20, 120, 40))
TEAM2_COLOR_LB = TEAM2_COLOR-TEAM2_COLOR_RANGE
TEAM2_COLOR_UB = TEAM2_COLOR+TEAM2_COLOR_RANGE

def save_templates():
    img = cv2.imread('img/hanamura/hanamura_660.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_locked.jpg', utils.crop(img, ICON_RECT_1))

    img = cv2.imread('img/hanamura/hanamura_4440.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_captured.jpg', utils.crop(img, ICON_RECT_1))

    img = cv2.imread('img/hanamura/hanamura_3120.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_A_1.jpg', utils.crop(img, ICON_RECT_1))
    img = cv2.imread('img/hanamura/hanamura_3390.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_A_cap_1.jpg', utils.crop(img, ICON_RECT_1))

    img = cv2.imread('img/hanamura/hanamura_11130.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_A_2.jpg', utils.crop(img, ICON_RECT_1))
    img = cv2.imread('img/hanamura/hanamura_23370.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_A_cap_2.jpg', utils.crop(img, ICON_RECT_1))

    img = cv2.imread('img/hanamura/hanamura_5040.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_B_1.jpg', utils.crop(img, ICON_RECT_2))
    img = cv2.imread('img/hanamura/hanamura_6990.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_B_cap_1.jpg', utils.crop(img, ICON_RECT_2))

    img = cv2.imread('img/hanamura/hanamura_15600.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_B_2.jpg', utils.crop(img, ICON_RECT_2))
    img = cv2.imread('img/hanamura/hanamura_17370.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_B_cap_2.jpg', utils.crop(img, ICON_RECT_2))

def read_tempaltes():
    templates = {}

    img_locked = cv2.imread('template/assult_locked.jpg', cv2.IMREAD_COLOR)
    mask_locked = np.zeros((img_locked.shape[0],img_locked.shape[1]), dtype=np.uint8)
    mask_locked = cv2.fillConvexPoly(mask_locked, ICON_MASK, 255)
    templates['locked'] = (img_locked, mask_locked)

    img_captured = cv2.imread('template/assult_captured.jpg', cv2.IMREAD_COLOR)
    mask_captured= np.zeros((img_captured.shape[0],img_captured.shape[1]), dtype=np.uint8)
    mask_captured = cv2.fillConvexPoly(mask_captured, ICON_MASK, 255)
    templates['captured'] = (img_captured, mask_captured)

    for point in ['A', 'B']:
        templates[point] = {}
        for team in [1,2]:
            img = cv2.imread('template/assult_'+point+'_'+str(team)+'.jpg', cv2.IMREAD_COLOR)
            mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
            mask = cv2.fillConvexPoly(mask, ICON_MASK, 255)
            templates[point][team] = (img, mask)

    for point in ['A', 'B']:
        templates[point+'cap'] = {}
        for team in [1,2]:
            img = cv2.imread('template/assult_'+point+'_cap_'+str(team)+'.jpg', cv2.IMREAD_COLOR)
            mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
            mask = cv2.fillConvexPoly(mask, ICON_MASK, 255)
            templates[point+'cap'][team] = (img, mask)

    # img_locked[mask_locked == 0] = (0,0,0)
    # img_captured[mask_captured == 0] = (0,0,0)
    # temp, mask = templates['Bcap'][1]
    # temp[mask == 0] = (0,0,0)
    # cv2.imshow('locked with mask', img_locked)
    # cv2.imshow('captured with mask', img_captured)
    # cv2.imshow('temp with mask', temp)
    # cv2.waitKey(0)

    return templates

def read_status(src, templates):
    status = {}

    for point in ['A', 'B']:
        if point == 'A':
            progress_rect = PROGRESS_RECT_1
        else:
            progress_rect = PROGRESS_RECT_2

        img = utils.crop(src, progress_rect)

        scores = []
        template, mask = templates['locked']
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        loc = np.unravel_index(np.argmax(res), res.shape)
        scores.append((-1, None, loc, res[loc]))

        template, mask = templates['captured']
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        loc = np.unravel_index(np.argmax(res), res.shape)
        scores.append((0, None, loc, res[loc]))

        for team in [1,2]:
            template, mask = templates[point][team]
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            loc = np.unravel_index(np.argmax(res), res.shape)
            scores.append((team, False, loc, res[loc])) # defender team

        # NOTE: May mismatch not capping
        for team in [1,2]:
            template, mask = templates[point+'cap'][team]
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            loc = np.unravel_index(np.argmax(res), res.shape)
            scores.append((team, True, loc, res[loc])) # defender team

        scores.sort(reverse=True, key=lambda m:m[3])
        score = scores[0]
        # print(point, scores)
        if score[3] > ICON_THRESHOLD:
            status[point] = (score[0], score[1], (score[2][1],score[2][0]))
        else:
            status[point] = (None, None, None)

    return status['A'][0],status['A'][1],status['A'][2],status['B'][0],status['B'][1],status['B'][2]

def read_point(img_progress, center, team):
    mask= np.zeros((img_progress.shape[0],img_progress.shape[1]), dtype=np.uint8)
    mask = cv2.circle(mask, center, 16, 255, thickness=4)
    img_progress[mask == 0] = (0,0,0)

    if team == 1:
        img_match = utils.match_color(img_progress, TEAM2_COLOR_LB, TEAM2_COLOR_UB)
    else:
        img_match = utils.match_color(img_progress, TEAM1_COLOR_LB, TEAM1_COLOR_UB)
    img_match = cv2.morphologyEx(img_match, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations=1, borderValue=0)

    # Convert all valid pixles to angle
    pts = np.transpose(np.array((img_match > 0).nonzero()))
    pts[:,[0,1]] = pts[:,[1,0]] # Swap so that pt=(x,y)
    # Calculate angle using atan2(y,x)
    # bottom to top is x, left to right is y
    theta = np.arctan2(pts[:,0]-center[0], -(pts[:,1]-center[1]))
    # Convert to 0-2pi
    theta[theta < 0] += 2*np.pi
    # Sort and covnert to degrees
    theta = np.sort(theta)/np.pi*180
    if theta.shape[0] > 1:
        # Use average delta theta and initial angle to filter miss match
        dtheta = theta[1:]-theta[0:-1]
        if np.mean(dtheta) < 5 and theta[0] < 15:         # Mean of dtheta indicates the scatterness of data
            start = -1
            indices = (dtheta > 20).nonzero()[0] # If big gap, use value before gap
            if len(indices) > 0: start = indices[0]
            progress = round(theta[start]/360*100)
            return progress

    return None

def read_progress(src, templates):
    status_a, cap_a, loc_a, status_b, cap_b, loc_b = read_status(src, templates)

    img_full_progress = utils.crop(src, PROGRESS_RECT)
    if status_a is None or status_b is None: return None, None, None
    # if point a is locked, point b must be locked
    if status_a == -1: return None, -1, -1
    if status_a == 0 and status_b == 0: return None, 100, 100
    if status_a == 0 and status_b == -1: return None, 100, -1

    progress = {'A': 0, 'B': 0}
    cap = {'A':cap_a, 'B':cap_b}
    loc = {'A': loc_a, 'B': loc_b}
    team = {'A': status_a, 'B': status_b} # Status indicates defending team, 1 or 2
    progress_rect = {'A': PROGRESS_RECT_1, 'B': PROGRESS_RECT_2}
    if status_a > 0:
        # Must be capturing point a
        progress['B'] = -1
        point = 'A'
    else:
        # Must be capturing point b
        progress['A'] = 100
        point = 'B'

    if not cap[point]:
        progress[point] = None
        return team[point], progress['A'], progress['B']

    # Overtime will offset in y direction
    # print(loc[point][1])
    if loc[point][1] > 11:
        dy = 14
    else:
        dy = 10
    img_progress = utils.crop(src, progress_rect[point])
    center = (int(progress_rect[point][2]/2), int(progress_rect[point][3]/2+(dy-ICON_RECT_1[1]+progress_rect[point][1])))
    percent = read_point(img_progress, center, team[point])
    if percent is not None: progress[point] = percent

    return team[point], progress['A'], progress['B']

save_templates()
templates = read_tempaltes()

def process_status(src):
    status_a, cap_a, loc_a, status_b, cap_b, loc_b = read_status(src, templates)
    img = utils.crop(src, PROGRESS_RECT)

    return '{} {:.2} {} {:.2}'.format(
        utils.val_to_string(status_a),
        utils.val_to_string(cap_a),
        utils.val_to_string(status_b),
        utils.val_to_string(cap_b),
    ), img

def mark_progress(img, progress, dx, dy):
    if progress is not None and progress > -1 and progress < 100:
        x_center = dx
        y_center = dy
        rad = progress/100*np.pi*2
        x = int(x_center+np.sin(rad)*16)
        y = int(y_center-np.cos(rad)*16)
        img = cv2.circle(img, (x,y), 3, 0, thickness=-1)

    return img

def process_progress(src):
    team, progress_A, progress_B = read_progress(src, templates)
    img_full_progress = utils.crop(src, PROGRESS_RECT)

    mark_progress(
        img_full_progress,
        progress_A,
        PROGRESS_RECT_1[0]-PROGRESS_RECT[0]+PROGRESS_RECT_1[2]/2,
        PROGRESS_RECT_1[3]/2,
    )

    mark_progress(
        img_full_progress,
        progress_B,
        PROGRESS_RECT_2[0]-PROGRESS_RECT[0]+PROGRESS_RECT_2[2]/2,
        PROGRESS_RECT_1[3]/2,
    )

    return '{} {} {}'.format(
        utils.val_to_string(team),
        utils.val_to_string(progress_A),
        utils.val_to_string(progress_B)
    ), img_full_progress

def save(start, end, code):
    obj = {
        'type':'assult',
        'status': [],
        'progress': {
            'A': [],
            'B': []
        }
    }

    for src, frame in utils.read_frames(start=start, end=end, code=code):
        team, progress_A, progress_B = read_progress(src, templates)
        obj['status'].append(team)
        obj['progress']['A'].append(progress_A)
        obj['progress']['B'].append(progress_B)
        print('Frame {:d} analyzed'.format(frame))

    utils.save_data('obj', obj, start, end, code)

def refine(code):
    obj = utils.load_data('obj',0,None,code)

    # obj['status'] = utils.remove_outlier(obj['status'],2)
    # for point in ['A', 'B']:
    #     obj['progress'][point] = utils.remove_outlier(obj['progress'][point],2,['none','number'])
    #     obj['progress'][point] = utils.remove_outlier(obj['progress'][point],3,['change'])
    #     obj['progress'][point] = utils.remove_outlier(obj['progress'][point],2,['change'])

    plt.figure('obj')
    plt.subplot(3,1,1)
    plt.plot(obj['status'])
    plt.subplot(3,1,2)
    plt.plot(obj['progress']['A'])
    plt.subplot(3,1,3)
    plt.plot(obj['progress']['B'])
    plt.show()

# utils.read_batch(process_status, map='hanamura', length=1623, start=1, num_width=12, num_height=16)
# utils.read_batch(process_progress, map='hanamura', length=1623, start=6, num_width=12, num_height=16)
# save(0, None, 'hanamura')
refine('hanamura')
