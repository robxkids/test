import cv2
import numpy

video = cv2.VideoCapture('outpy.avi')

while(video.isOpened()):
    ret, frame = video.read()
    if ret:


        cv2.imshow('color', frame)


    else:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('screen.png', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()