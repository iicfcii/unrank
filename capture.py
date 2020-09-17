import cv2

vidName = 'overwatch_1_1'
vid = cv2.VideoCapture('vid/'+vidName+'.mp4')

fps = int(vid.get(cv2.CAP_PROP_FPS))
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps, width, height)


frame_count = 0
while True:
    ret, frame = vid.read()

    if ret:
        if frame_count%(fps/1) == 0: # 1 image per second
            imgName = 'img/'+vidName+'_'+str(frame_count)+'.jpg'
            print(imgName)
            cv2.imwrite(imgName, frame);
        frame_count = frame_count + 1
    else:
        break
