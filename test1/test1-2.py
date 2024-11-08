import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_img(_img):
    cv2.imshow('image', _img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像的梯度 sobel因子
# 作用：获取图像的轮廓
def img_gradient_sobel():
    img = cv2.imread('../img/pie.png')
    # 图片不干净，先去噪点
    img2 = cv2.medianBlur(img, 3)
    # 让不是0的黑色点，变成0
    ret, img3 = cv2.threshold(img2, 254, 255, cv2.THRESH_TOZERO)

    # ddepth: 输出图像的深度，-1表示与输入图像相同 cv2.CV_64F
    # dx,dy 水平和竖直方向
    # dx 右边-左边
    # ksize Sobel的算子大小
    dst = cv2.Sobel(img3, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    # 由于得出来得负数会被截断成0，所以要取绝对值
    dst = cv2.convertScaleAbs(dst)

    # dy方向下边-上边
    dst2 = cv2.Sobel(img3, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    dst2 = cv2.convertScaleAbs(dst2)

    # 求和
    sobelxy = cv2.addWeighted(dst, 0.5, dst2, 0.5, 0)
    show_img(sobelxy)


# 图像的梯度 Scharr因子 核中数据更大 捕捉更丰富的信息
# laplacian因子 会被噪点影响，四周-中间
def img_gradient_scharr_laplacian():
    img = cv2.imread('../img/pie.png')
    # 图片不干净，先去噪点
    img2 = cv2.medianBlur(img, 3)
    # 让不是0的黑色点，变成0
    ret, img3 = cv2.threshold(img2, 254, 255, cv2.THRESH_TOZERO)

    # ddepth: 输出图像的深度，-1表示与输入图像相同 cv2.CV_64F
    # dx,dy 水平和竖直方向
    # dx 右边-左边
    dst = cv2.Scharr(img3, ddepth=cv2.CV_64F, dx=1, dy=0)
    # 由于得出来得负数会被截断成0，所以要取绝对值
    dst = cv2.convertScaleAbs(dst)

    # dy方向下边-上边
    dst2 = cv2.Scharr(img3, ddepth=cv2.CV_64F, dx=0, dy=1)
    dst2 = cv2.convertScaleAbs(dst2)

    # 求和
    sobelxy = cv2.addWeighted(dst, 0.5, dst2, 0.5, 0)

    dst3 = cv2.Laplacian(img3, ddepth=cv2.CV_64F)
    dst3 = cv2.convertScaleAbs(dst3)

    res = np.hstack((sobelxy, dst3))
    show_img(res)


# Canny边缘检测
def img_canny():
    img = cv2.imread('../img/cactus.png', cv2.IMREAD_GRAYSCALE)
    # 使用高斯滤波 平滑图像 滤除噪点
    # 计算图像每一个像素点的梯度强度和方向
    # 应用非极大值抑制， 消除边缘检测带来的杂散响应
    # 应用双阈值检测来确定真实和潜在的边缘
    # 通过抑制孤立的弱边缘最终完成边缘检测 指定越大，检测越明显的边缘 越小，会检测到细微的边缘
    v1 = cv2.Canny(img, 80, 150)
    v2 = cv2.Canny(img, 50, 100)
    res = np.hstack((v1, v2))
    show_img(res)


img_canny()
