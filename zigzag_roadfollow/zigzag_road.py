import cv2
import numpy as np

video = cv2.VideoCapture('outpy.avi')

while(video.isOpened()):
    ret, frame = video.read()
    if ret:
        # бинаризация изображения с выделением желтой и белой разметки отдельно
        # желтая линия разметки: синий(0 - 130), зеленый (130 - 255), красный (140 - 255)
        # белая линия разметки: синий(140 - 255), зеленый (120 - 255), красный (120 - 255)
        binary_yellow = cv2.inRange(frame, (0, 130, 140), (130, 255, 255))
        binary_white = cv2.inRange(frame, (140, 120, 120), (255, 255, 255))

        # преобразование перспективы, выравнивание дорожной разметки, которая уходит в даль ...
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

        # создаем изображение с 3-мя цветовыми каналами для визуализации
        out_yellow = np.dstack((warped_yellow, warped_yellow, warped_yellow))
        out_white = np.dstack((warped_white, warped_white, warped_white))
        out_img = out_white + out_yellow

        # строим гистограммы (линии с наибольшим кол-вом белым пикселей) для каждой полосы разметки отдельно
        histogram_yellow = np.sum(warped_yellow[:,:], axis=0) # histogram_yellow = [1520, 0, 255 ...  512] len(histogram_yellow) = 400
        midpoint = histogram_yellow.shape[0] // 2 # 200
        indWhiteColumnL = np.argmax(histogram_yellow[:midpoint]) # индекс элемента с наибольшим значением
        warped_yellow_visual = warped_yellow.copy()

        histogram_white = np.sum(warped_white[:,:], axis=0)
        midpoint2 = histogram_white.shape[0] // 2
        indWhiteColumnR = np.argmax(histogram_white[midpoint2:]) + midpoint2
        warped_white_visual = warped_white.copy()

        # создаем окна вдоль оси гистограммы
        nwindows = 9
        window_height = out_img.shape[0] // 9
        window_half_width = 40
        XCenterLeftWindow = indWhiteColumnL
        XCenterRightWindow = indWhiteColumnR

        # создаем два пустых массива в которых будем хранить пиксели линий которые попали внутрь окна
        left_lane_inds = np.array([], dtype=np.int16)
        right_lane_inds = np.array([], dtype=np.int16)

        # выделяем координаты белых пикселей на бинаризованных изображениях линий
        nonzero_yellow = warped_yellow.nonzero() # возвращает индексы где лежат НЕчерные пиксели
        # пример работы .nonzero()
        # l = [[255, 255, 0, 255],
        #      [255, 255, 0, 0]]
        # l[0][0] 255
        # l[0][1] 255
        #     -------
        # d = l.nonzero(0)
        # d = array([0, 0, 0, 1, 1]), array([0, 1, 3, 0, 1])


        WhitePixelIndY_yellow = np.array(nonzero_yellow[0])
        WhitePixelIndX_yellow = np.array(nonzero_yellow[1])

        nonzero_white = warped_white.nonzero()
        WhitePixelIndY_white = np.array(nonzero_white[0])
        WhitePixelIndX_white = np.array(nonzero_white[1])

        # считаем координаты окон для каждой оси отдельно
        for window in range(nwindows):
            win_y1 = out_img.shape[0] - (window + 1) * window_height
            win_y2 = out_img.shape[0] - (window) * window_height
            left_win_x1 = XCenterLeftWindow - window_half_width
            left_win_x2 = XCenterLeftWindow + window_half_width
            right_win_x1 = XCenterRightWindow - window_half_width
            right_win_x2 = XCenterRightWindow + window_half_width

            # рисуем окна по координатам для левой и правой части разметки раздельно
            cv2.rectangle(out_img, (left_win_x1, win_y1), (left_win_x2, win_y2), (0, 50 + window * 24, window * 28), 2)
            cv2.rectangle(out_img, (right_win_x1, win_y1), (right_win_x2, win_y2), (0, 50 + window * 24, 0), 2)

            # найдем белые пиксели которые попали внутрь окон
            good_yellow_inds = ((WhitePixelIndY_yellow >= win_y1) & (WhitePixelIndY_yellow <= win_y2)
                                & (WhitePixelIndX_yellow >= left_win_x1) & (WhitePixelIndX_yellow <= left_win_x2)).nonzero()[0]
            good_white_inds = ((WhitePixelIndY_white >= win_y1) & (WhitePixelIndY_white <= win_y2)
                                & (WhitePixelIndX_white >= right_win_x1) & (WhitePixelIndX_white <= right_win_x2)).nonzero()[0]

            # добавляем белые пиксели которые попали внутрь окна в общий массив
            left_lane_inds = np.concatenate((left_lane_inds, good_yellow_inds))
            right_lane_inds = np.concatenate((right_lane_inds, good_white_inds))

            # если внутрь окна попало более 50 "хороших" пикселей, смещаем след. окна на среднюю X координату
            # этих пикселей
            if len(good_yellow_inds) > 50:
                XCenterLeftWindow = np.int(np.mean(WhitePixelIndX_yellow[good_yellow_inds]))
            if len(good_white_inds) > 50:
                XCenterRightWindow = np.int(np.mean(WhitePixelIndX_white[good_white_inds]))

        # раскрашиваем пиксели которые попали внутрь окон
        out_img[WhitePixelIndY_yellow[left_lane_inds], WhitePixelIndX_yellow[left_lane_inds]] = [0, 255, 255]
        out_img[WhitePixelIndY_white[right_lane_inds], WhitePixelIndX_white[right_lane_inds]] = [0, 255, 0]
        # короткая домашка - раскрасить аналогично пиксели правой разметки (белой)

        leftx = WhitePixelIndX_yellow[left_lane_inds]
        lefty = WhitePixelIndY_yellow[left_lane_inds]
        rightx = WhitePixelIndX_white[right_lane_inds]
        righty = WhitePixelIndY_white[right_lane_inds]

        if leftx.size and lefty.size and rightx.size and righty.size:
            left_fit = np.polyfit(lefty, leftx, 2)
            # [ 1.86287301e-03 (a) -1.25976411e+00 (b)  1.34310725e+02 (c)]
            right_fit = np.polyfit(righty, rightx, 2)
            # print(left_fit)

            for ver_ind in range(out_img.shape[0]):
                # y = ax2 + bx + c
                gor_ind = left_fit[0] * (ver_ind ** 2) + left_fit[1] * ver_ind + left_fit[2]
                gor_ind_w = right_fit[0] * (ver_ind ** 2) + right_fit[1] * ver_ind + right_fit[2]
                cv2.circle(out_img, (int(gor_ind), int(ver_ind)), 2, (255, 0, 255), 1)
                cv2.circle(out_img, (int(gor_ind_w), int(ver_ind)), 2, (255, 0, 255), 1)
            # задание на дом: посчитать и нарисовать линию (траекторию) по которой должна двигаться машинка, т.е.
            # между левой и правой полосой разметки

        # выводим полученные изображения на экран
        cv2.imshow("frame", frame)
        cv2.imshow("warped_yellow", warped_yellow_visual)
        cv2.imshow("warped_white", warped_white_visual)
        cv2.imshow("out", out_img)


    else:
        # перематываем видео на 0-й кадр если оно закончилось
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # горячие клавиши для создания скриншота и остановки программы
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('screen.png', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()

