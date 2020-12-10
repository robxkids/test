import cv2
import numpy as np

video = cv2.VideoCapture('outpy.avi')

while(video.isOpened()):
    ret, frame = video.read()
    if ret:
        #желтая линия разметки: синий(0 - 130), зеленый (130 - 255), красный (140 - 255)
        #белая линия разметки: синий(140 - 255), зеленый (120 - 255), красный (120 - 255)
        binary_yellow = cv2.inRange(frame, (0, 130, 140), (130, 255, 255))
        binary_white = cv2.inRange(frame, (140, 120, 120), (255, 255, 255))

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        src = np.int32([[0, frame_height - 50], [50, frame_height/3], [frame_width - 50, frame_height/3], [frame_width, frame_height - 50]])
        src_float = np.array(src, dtype=np.float32)
        cv2.polylines(frame, [src], True, (255, 0, 0), 5)

        dst_height = int((frame_height - 50 - frame_height/3)/2)
        dst_width = int(frame_width/2)
        dst = np.float32([[0, dst_height], [0, 0], [dst_width, 0], [dst_width, dst_height]])
        M = cv2.getPerspectiveTransform(src_float, dst)

        warped_yellow = cv2.warpPerspective(binary_yellow, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR)
        warped_white = cv2.warpPerspective(binary_white, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR)

        cv2.imshow("frame", frame)
        cv2.imshow("warped_yellow", warped_yellow)
        cv2.imshow("warped white", warped_white)

    else:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('screen.png', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()