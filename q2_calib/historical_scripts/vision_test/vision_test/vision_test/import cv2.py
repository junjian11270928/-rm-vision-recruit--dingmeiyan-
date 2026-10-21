import cv2
import numpy as np

# ========== 参数（角点数量！）==========
corners_x = 8   # 横向内部角点数
corners_y = 6   # 纵向内部角点数
square_pix = 200 # 每个方格像素尺寸
# ======================================

w = (corners_x + 1) * square_pix
h = (corners_y + 1) * square_pix
img = np.zeros((h, w), dtype=np.uint8)

for y in range(corners_y+1):
    for x in range(corners_x+1):
        if (x + y) % 2 == 0:
            ys = y * square_pix
            ye = ys + square_pix
            xs = x * square_pix
            xe = xs + square_pix
            img[ys:ye, xs:xe] = 255

cv2.imwrite("chessboard.png", img)
cv2.imshow("board", img)
cv2.waitKey(0)
cv2.destroyAllWindows()