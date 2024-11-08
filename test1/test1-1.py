import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_img(_img):
    cv2.imshow('image', _img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_video():
    # 读取视频
    video = cv2.VideoCapture('../video/1.mp4')
    # 检查是否打开
    _open = video.isOpened()

    while _open:
        _open, frame = video.read()
        if _open is False:
            break
        # 显示视频帧
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        # 按下 q 键退出
        if cv2.waitKey(10) & 0xFF == 27:
            break
    video.release()
    cv2.destroyAllWindows()


def read_img():
    # 读取通道 BGR
    img = cv2.imread('../img/dog.png')
    # 读取灰度图
    # img2 = cv2.imread('../img/dog.png', cv2.IMREAD_GRAYSCALE)
    # show_img(img2)
    # 图像保存
    # cv2.imwrite('../img/dog_gray.png', img2)
    # print(img2.shape)

    # 截取图像
    # img_crop = img[0:200, 0:100]
    # show_img(img_crop)
    # print(img_crop.shape)

    # 颜色通道提取
    b, g, r = cv2.split(img)
    # print(b)

    # 组合rgb
    img_merge_plt = cv2.merge([r, g, b])
    # show_img(img_merge)

    # 只保留单一通道R
    # img_cy = img.copy()
    # img_cy[:, :, 0] = 0
    # img_cy[:, :, 1] = 0
    # show_img(img_cy)

    # 图像填充
    top_size, left_size, bottom_size, right_size = (50, 50, 50, 50)
    # constant 常量填充，填充指定的颜色
    img_pad1 = cv2.copyMakeBorder(img_merge_plt, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT,
                                  value=[255, 255, 255])
    # replicate 复制最边缘的像素
    img_pad2 = cv2.copyMakeBorder(img_merge_plt, top_size, bottom_size, left_size, right_size, cv2.BORDER_REPLICATE)
    # reflect 反射法，对边缘图像进行反向复制    ytrewq|qwertyuiopasdfghjk
    img_pad3 = cv2.copyMakeBorder(img_merge_plt, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
    # reflect101 反射法101，以边缘像素为轴与reflect有一个像素之差   ytrew|qwertyuiopasdfghjk
    img_pad4 = cv2.copyMakeBorder(img_merge_plt, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
    # wrap 外包装法 取对边的像素复制  fghjk|qwertyuiopasdfghjk
    img_pad5 = cv2.copyMakeBorder(img_merge_plt, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)

    #### 由于plt中使用的通道是rbg，所以使用重组后的图片img_merge_plt

    # subplot 同一个图形窗口中创建多个子图
    # plt.subplot(2, 1, 1)  # 2行1列，当前是第1个子图
    plt.subplot(231), plt.imshow(img_merge_plt), plt.title('origin')
    plt.subplot(232), plt.imshow(img_pad1, 'gray'), plt.title('constant')
    plt.subplot(233), plt.imshow(img_pad2, 'gray'), plt.title('replicate')
    plt.subplot(234), plt.imshow(img_pad3, 'gray'), plt.title('reflect')
    plt.subplot(235), plt.imshow(img_pad4, 'gray'), plt.title('reflect101')
    plt.subplot(236), plt.imshow(img_pad5, 'gray'), plt.title('wrap')
    plt.show()


def img_number():
    # 数值计算
    img = cv2.imread('../img/dog.png')
    img2 = img + 10
    # 超过255的会 %255，因为类型是dtype=uint8
    show_img(img2)
    # <class 'numpy.uint8'>
    # print(type(img2[0,0,0]))
    # 与直接相加不同，这个会在大于255的时候只取255
    img3 = cv2.add(img, 10)
    show_img(img3)


def merge_img():
    # 图像融合
    img1 = cv2.imread('../img/dog.png')
    img2 = cv2.imread('../img/cat.png')

    # 先将两张图片的尺寸给统一
    print(img1.shape)
    print(img2.shape)

    # resize(img,(width,height))
    # 图片的尺寸是(height,width,channel)
    img3 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    # 直接相加效果不好
    # img4 = img1 + img3
    # 按照比例相加，但是会有小数
    # img4 = 0.5 * img1 + 0.5 * img3
    # 使用cv2.addWeighted 自动四舍五入还可以加上偏置
    # img4 = cv2.addWeighted(img1, 0.8, img3, 0.5, 0)
    # show_img(img4)

    # resize(img,(width,height), 放大倍数)
    img5 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
    show_img(img5)


# 图像阈值处理
def img_thresh():
    img = cv2.imread('../img/dog.png')
    b, g, r = cv2.split(img)
    # 组合rgb
    img_merge_plt = cv2.merge([r, g, b])
    # ret, dst = cv2.threshold(src, thresh, maxval, type)
    # dst: 输出图像
    # src: 输入图像 只能输入单通道图像，通常输入灰度图
    # thresh: 阈值
    # maxval: 图像中高于（或低于）阈值的像素点的最大值，根据type决定
    # type-: 阈值类型 包括物种类型
    #   cv2.THRESH_BINARY: 阈值二值化，大于阈值是maxval，小于是0
    #   cv2.THRESH_BINARY_INV: 阈值二值化，大于阈值是0，小于是maxval
    #   cv2.THRESH_TRUNC: 截断，大于阈值是maxval，否则不变
    #   cv2.THRESH_TOZERO: 截断，小于阈值是0，否则不变
    #   cv2.THRESH_TOZERO_INV: 截断，大于阈值是0，否则不变
    ret, thresh1 = cv2.threshold(img_merge_plt, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img_merge_plt, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img_merge_plt, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img_merge_plt, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img_merge_plt, 127, 255, cv2.THRESH_TOZERO_INV)

    title = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img_merge_plt, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        # TODO 为什么不是灰色呢
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(title[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


# 图像平滑处理
def img_blur():
    img = cv2.imread('../img/house.png')
    # 均值滤波
    blur_img1 = cv2.blur(img, (3, 3))
    # 方框滤波 归一化 与均值滤波一致
    blur_img2 = cv2.boxFilter(img, -1, (3, 3), normalize=True)
    # 方框滤波 不归一化 越界之后 显示 255
    blur_img3 = cv2.boxFilter(img, -1, (3, 3), normalize=False)

    # 高斯滤波 离得越近 权重越大
    blur_img4 = cv2.GaussianBlur(img, (5, 5), 0)
    # 中值滤波 找到区域内中间的值 能很好的处理噪点
    blur_img5 = cv2.medianBlur(img, 7)

    # 工具方法，多张图片并排展示
    res = np.hstack((blur_img1, blur_img5))
    show_img(res)


# 腐蚀操作 腐蚀掉边缘 一般用于二值操作（黑或白） 去毛刺
def img_erode():
    img = cv2.imread('../img/cactus.png')
    kernel = np.ones((3, 3), np.uint8)
    erode_img = cv2.erode(img, kernel, iterations=1)
    show_img(erode_img)


# 膨胀操作
def img_dilate():
    img = cv2.imread('../img/dog.png')
    kernel = np.ones((3, 3), np.uint8)
    dilate_img = cv2.dilate(img, kernel, iterations=2)
    show_img(dilate_img)


# 开运算和闭运算
def img_open_close():
    img = cv2.imread('../img/dog.png')
    kernel = np.ones((3, 3), np.uint8)
    # 开 先腐蚀再膨胀
    open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    show_img(open_img)
    # 闭 先膨胀再腐蚀
    close_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    show_img(close_img)


# 梯度运算 膨胀-腐蚀 获得轮廓信息
def img_gradient():
    img = cv2.imread('../img/dog.png')
    kernel = np.ones((3, 3), np.uint8)
    gradient_img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    show_img(gradient_img)


# 礼帽与黑帽 礼帽：原图-开运算，黑帽：闭运算-原图
def img_hat():
    img = cv2.imread('../img/dog.png')
    kernel = np.ones((3, 3), np.uint8)
    hat_img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    show_img(hat_img)
    hat_img2 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    show_img(hat_img2)


img_hat()
