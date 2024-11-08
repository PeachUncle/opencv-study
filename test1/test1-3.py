import cv2


def show_img(_img):
    cv2.imshow('image', _img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像轮廓