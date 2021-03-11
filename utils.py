import cv2
import numpy as np
import os
import json

DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'data')
IMG_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'img')

def val_to_string(val):
    if val is None: return 'NA'
    return str(val)

def offset_rect(rect, dx, dy):
    rect = list(rect)
    rect[0] += dx
    rect[1] += dy
    return tuple(rect)

def crop(img, rect):
    x, y, w, h = rect

    if x < 0: x = 0
    if y < 0: y = 0
    if x+w > img.shape[1]: w = img.shape[1]-x
    if y+h > img.shape[0]: h = img.shape[0]-y

    img_crop = img[y:y+h,x:x+w]
    return img_crop.copy()

def pad_rect(rect, dx, dy):
    return (
        rect[0]-dx,
        rect[1]-dy,
        rect[2]+dx*2,
        rect[3]+dy*2
    )


# HSV color space
def match_color(img, lb, ub):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_bin = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)

    if ub[0] > 180: # Red is across 180
        h_match = np.logical_or(img_hsv[:,:,0] > lb[0], img_hsv[:,:,0] < ub[0]-180)
    else:
        h_match = np.logical_and(img_hsv[:,:,0] > lb[0], img_hsv[:,:,0] < ub[0])
    s_match = np.logical_and(img_hsv[:,:,1] > lb[1], img_hsv[:,:,1] < ub[1])
    v_match = np.logical_and(img_hsv[:,:,2] > lb[2], img_hsv[:,:,2] < ub[2])

    img_bin[np.all((h_match, s_match, v_match), axis=0)] = 255

    return img_bin

def extract_locs(res, match_threshold, dist_threshold):
    locs = np.where(res>=match_threshold)

    if len(locs[0]) == 0: return []

    # Sort locations so that ones with higher res are preserved
    # locations:[(x,y,score),(x,y,score),(x,y,score)...]
    locs = [(x, y, res[y,x]) for y,x in np.transpose(np.array(locs))]
    locs.sort(reverse=True, key=lambda l:l[2])

    # Clean up clustered location
    locs_s = []
    for l in locs:
        tooClose = False
        for l_s in locs_s:
            if (l[0]-l_s[0])**2+(l[1]-l_s[1])**2 < dist_threshold**2:
                tooClose = True
                break
        if not tooClose:
            locs_s.append(l)

    # print(locs_s)

    return locs_s

# When type is none
# min: minimum value the number can be
# max: maximum value the number can be
# When type is change
# min: minimum value when removing upward sudden change
# max: maximum value when removing downward sudden change
def remove_outlier(src, size=1, types=['none','number','change'], threshold=0.4, min=None, max=None, duration=0, interp=True):
    data = np.array(src)

    def remove(type):
        for i in range(len(data)-size-1):
            if ( # number None number
                type == 'none' and
                np.any(data[i+1:i+size+1] == np.array([None])) and
                data[i] is not None and
                data[i+size+1] is not None
            ):
                if interp:
                    if min is not None and (data[i] < min or data[i+size+1] < min):
                        continue

                    if max is not None and (data[i] > max or data[i+size+1] > max):
                        continue

                    data[i+1:i+size+1] = (data[i]+data[i+size+1])/2
                else:
                    data[i+1:i+size+1] = data[i]

            if ( # None number None
                type == 'number' and
                np.any(data[i+1:i+size+1] != np.array([None])) and
                data[i] is None and
                data[i+size+1] is None
            ):
                data[i+1:i+size+1] = None

            if ( # small Large small
                (type == 'change' or type == 'up' or type == 'down') and
                np.all(data[i:i+size+2] != np.array([None])) # All numbers
            ):
                if (
                    min is not None and
                    (type == 'up' or type=='change') and
                    (
                        np.all(data[i:i+size+2] < min) or
                        np.sum(data[i:i+size+2] >= min) < duration
                    )
                ):
                    continue

                if (
                    max is not None and
                    (type == 'down' or type=='change') and
                    (
                        np.all(data[i:i+size+2] > max) or
                        np.sum(data[i:i+size+2] <= max) < duration
                    )

                ):
                    continue

                d = data[i+1:i+size+2]-data[i:i+size+1]
                d_total = np.sum(np.absolute(d))
                d_net = np.absolute(np.sum(d))
                if d_net < d_total*threshold:
                    # Get max 2 absolute values and use the one with smaller index
                    # to determine sudden change direction
                    up = d[np.amin(np.argsort(np.abs(d))[-2:])] > 0
                    if (type == 'up' and up) or (type == 'down' and not up) or type =='change':
                        if interp:
                            data[i+1:i+size+1] = (data[i]+data[i+size+1])/2
                        else:
                            data[i+1:i+size+1] = data[i]

    for type in types:
        remove(type)

    return data.tolist()

# type: left, right, both
# no extend for start and end nones
def extend_none(mask, datas, size=1, type='both'):
    if size == 0:
        for i in range(size, len(mask), 1):
            if mask[i] is None:
                for data in datas:
                    data[i] = None
        return

    if type == 'left' or type == 'both':
        for i in range(size, len(mask), 1):
            if (
                mask[i] is None and
                mask[i-1] is not None and
                not np.all(mask[i:] == np.array([None]))
            ):
                mask[i-size:i] = [None]*size
                for data in datas:
                    data[i-size:i] = [None]*size

    if type == 'right' or type == 'both':
        for i in range(len(mask)-size-1, -1, -1):
            if (
                mask[i] is None and
                mask[i+1] is not None and
                not np.all(mask[:i] == np.array([None]))
            ):
                mask[i+1:i+1+size] = [None]*size
                for data in datas:
                    data[i+1:i+1+size] = [None]*size

def fix_disconnect(code, data, value):
    obj = load_data('obj_r',0,None,code)
    hero = load_data('hero',0,None,code)

    # Clean up the source hero data a bit
    extend_none(obj['status'], [hero[str(p)] for p in range(1,13)], size=0)
    for player in range(1,13):
        # Remove sudden changes, loosen threshold because values are smaller
        hero[str(player)] = remove_outlier(hero[str(player)], size=3, threshold=0.5, interp=False)

    DISCONNECT_SIZE = 12 # NOTE: may need to tune
    # Find potential disconnects
    dcs = []
    i = 0
    while i < len(obj['status'])-DISCONNECT_SIZE-1:
        # Skip point locked or black screen
        if obj['status'][i] == -1 or obj['status'][i] is None:
            i += 1
            continue

        for player in range(1,13):
            hs = np.array(hero[str(player)]) # Convert to np array
            if np.all(hs[i:i+DISCONNECT_SIZE] == -1):
                start = i
                end = i+DISCONNECT_SIZE
                while end < len(hs):
                    if np.all(hs[i+DISCONNECT_SIZE:end] == -1):
                        end += 1
                    else:
                        end -= 1
                        break

                dcs.append((1 if player < 7 else 2, start, end))
                i = end - 1

        i += 1
    dcs = list(set(dcs)) # Remove duplicates

    # Find the player who disconnects
    dcs_r = []
    for dc in dcs:
        team, start, end = dc
        # Beginning and end of match does not count
        if start < 40: continue
        if start > len(obj['status'])-10: continue

        heroes_current = []
        heroes_prev = []
        for p in range((team-1)*6+1,team*6+1):
            heroes_current.append(hero[str(p)][start])
            heroes_prev.append(hero[str(p)][start-1])
        # print(team, heroes_prev, heroes_current)

        for i in range(6):
            # NOTE: Assume no hero switch happens exactly when someone leaves
            if heroes_prev[i] not in heroes_current:
                dcs_r.append(((team-1)*6+i+1, team, start, end)) # player, team, start, end
    dcs_r = list(set(dcs_r)) # Remove duplicates

    if len(dcs_r) > 0: print('Player disconnection happens', dcs_r)

    # Adjust data accordingly
    for dc in dcs_r:
        player, team, start, end = dc
        for i in range(start, end):
            for p in range((team-1)*6+1,team*6+1):
                if p < player:
                    data[str(p)][i] = data[str(p+1)][i]

                if p == player:
                    data[str(p)][i] = value

def read_batch(process, start=0, code='nepal', num_width=8, num_height=8):
    length = count_frames(code)
    shape = None
    for i in range(start,int(length/num_width/num_height)+1):
        imgs = []
        for j in range(i*num_width*num_height, i*num_width*num_height+num_width*num_height):
            img = cv2.imread('img/'+code+'/'+code+'_'+str(j)+'.png', cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros(shape, dtype=np.uint8)
            else:
                print(j)
                info, img = process(img)
                shape = img.shape

            if j < length:
                img = cv2.putText(img, info, (0,img.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0))
                img = cv2.putText(img, str(j), (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0))
            imgs.append(img)

        imgs_row = []
        for i in range(num_height):
            imgs_row.append(cv2.hconcat(imgs[i*num_width:i*num_width+num_width], num_width))

        img_final = cv2.vconcat(imgs_row, num_height)
        cv2.imshow('read', img_final)
        cv2.waitKey(0)

def data_path(code):
    path = os.path.join(DATA_FOLDER_PATH, code)

    if not os.path.isdir(path):
        os.mkdir(path)

    return path

def img_path(code):
    path = os.path.join(IMG_FOLDER_PATH, code)

    if not os.path.isdir(path):
        os.mkdir(path)

    return path

def file_path(type, start_frame, end_frame, code, ext='json'):
    return os.path.join(
        data_path(code),
        '{}_{}_{:d}_{:d}.{}'.format(code, type, start_frame, end_frame, ext)
    )

def count_frames(code):
    path = os.path.join(IMG_FOLDER_PATH, code)

    for root, dirs, files in os.walk(path):
        return len(files)

    return 0

def read_frames(start, end, code):
    i = start
    if end is None: end = count_frames(code)

    while i == start or i < end:
        frame = i
        img = cv2.imread('img/'+code+'/'+code+'_'+str(frame)+'.png', cv2.IMREAD_COLOR)
        assert img is not None

        i += 1
        yield img, frame

    return

def save_data(type, data, start, end, code):
    if end is None: end = count_frames(code)-1

    with open(file_path(type, start, end, code), 'w') as json_file:
        json.dump(data, json_file)

def load_data(type, start, end, code):
    if end is None: end = count_frames(code)-1

    if end < 1: return None

    try:
        with open(file_path(type, start, end, code)) as f:
            data = json.load(f)

        return data
    except FileNotFoundError:
        return None
