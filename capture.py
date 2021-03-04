import cv2
import sys
from subprocess import Popen, PIPE, STDOUT, run
import time

# NOTE: Manually create folder first
vidName = '20210303'

# proc = Popen(
#     './ffmpeg/ffmpeg -y -f gdigrab -r 30 -i title=Overwatch -c:v h264_nvenc -b:v 5M -vf scale=-1:720 ./vid/gibraltar.mp4',
#     stdout=PIPE, stdin=PIPE, stderr=PIPE
# )
# time.sleep(10)
# proc.communicate(input='q'.encode())

vid = cv2.VideoCapture('vid/'+vidName+'.mp4')

FPS = 30
WIDTH = 1280
HEIGHT = 720

fps = int(vid.get(cv2.CAP_PROP_FPS))
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps != 30:
    print('Fps not correct', fps)
if width != 1280 or height != 720:
    print('Dimension not correct', width, height)

frame_count = 0
while True:
    ret, frame = vid.read()

    if ret:
        if frame_count%(FPS/1) == 0: # 1 image per second
            imgName = 'img/'+vidName+'/'+vidName+'_'+str(frame_count)+'.jpg'
            print(imgName)
            cv2.imwrite(imgName, frame);
        frame_count = frame_count + 1
    else:
        break
