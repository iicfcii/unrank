import cv2
import numpy as np
import os
import json

DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'data')
IMG_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'img')

def val_to_string(val):
    if val is None: return 'NA'
    return str(val)

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

def remove_outlier(src, size=1, types=['none','number','change'], threshold=0.4):
    data = np.array(src)

    def remove(type):
        for i in range(len(data)-size):
            if ( # number None number
                type == 'none' and
                np.any(data[i+1:i+size+1] == None) and
                data[i] is not None and
                data[i+size+1] is not None
            ):
                data[i+1:i+size+1] = (data[i]+data[i+size+1])/2
            if ( # None number None
                type == 'number' and
                np.any(data[i+1:i+size+1] != None) and
                data[i] is None and
                data[i+size+1] is None
            ):
                data[i+1:i+size+1] = None
            if ( # small Large small
                (type == 'change' or type == 'up' or type == 'down') and
                np.all(data[i:i+size+2] != None) # All numbers
            ):
                d = data[i+1:i+size+2]-data[i:i+size+1]
                d_total = np.sum(np.absolute(d))
                d_net = np.absolute(np.sum(d))
                if d_net < d_total*threshold:
                    # Get max 2 absolute values and use the one with smaller index
                    up = d[np.amin(np.argsort(np.abs(d))[-2:])] > 0
                    if (type == 'up' and up) or (type == 'down' and not up) or type =='change':
                        data[i+1:i+size+1] = (data[i]+data[i+size+1])/2

    for type in types:
        remove(type)

    return data.tolist()

def read_batch(process, start=0, map='nepal', length=835, num_width=8, num_height=8):
    shape = None
    for i in range(start,int(length/num_width/num_height)+1):
        imgs = []
        for j in range(i*num_width*num_height, i*num_width*num_height+num_width*num_height):
            img = cv2.imread('img/'+map+'/'+map+'_'+str(j*30)+'.jpg', cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros(shape, dtype=np.uint8)
            else:
                print(j*30)
                info, img = process(img)
                shape = img.shape

            if j < length:
                img = cv2.putText(img, info, (0,img.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0))
                img = cv2.putText(img, str(j*30), (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0))
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

def file_path(type, start_frame, end_frame, code):
    return os.path.join(
        data_path(code),
        '{}_{}_{:d}_{:d}.json'.format(code, type, start_frame, end_frame)
    )

def count_frames(code='nepal'):
    path = os.path.join(IMG_FOLDER_PATH, code)
    return len(next(os.walk(path))[2])

def read_frames(start=0, end=None, code='nepal'):
    i = start
    if end is None: end = count_frames(code)

    while i == start or i < end:
        frame = i*30
        img = cv2.imread('img/'+code+'/'+code+'_'+str(frame)+'.jpg', cv2.IMREAD_COLOR)
        assert img is not None

        i += 1
        yield img, frame

    return

def save_data(type, data, start, end, code):
    if end is None: end = count_frames(code)-1

    with open(file_path(type, start*30, end*30, code), 'w') as json_file:
        json.dump(data, json_file)

def load_data(type, start, end, code):
    if end is None: end = count_frames(code)-1

    with open(file_path(type, start*30, end*30, code)) as f:
        data = json.load(f)

    return data
