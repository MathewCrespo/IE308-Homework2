import cv2
import  numpy as np
#R 通道
def R (pixel):
    if pixel<127:
        return 0
    elif pixel>191:
        return (pixel-20)
    else:
        return 4*pixel-510
# G通道
def G (pixel):
    if pixel<= 63:
        return 254-4*pixel
    elif (pixel >= 64 and pixel <=127):
        return (pixel-191)*4-254
    elif (pixel>=128 and pixel <=191):
        return 255
    else:
        return (1022-3*pixel)
# B通道
def B (pixel):
    if (0<=pixel and pixel<=63):
        return 255
    elif (64<=pixel and pixel<=127):
        return 510-4*pixel
    else:
        return 0

image_path = "F:\\anaconda\\Scripts\\lena.jpg"
img = cv2.imread(image_path, 0)
width, length = np.shape(img)
color = np.zeros((width,length,3), np.uint8)
for i in range(width):
    for j in range(length):
        color[i,j,2] = R(img[i][j])
        color[i,j,0] = B(img[i][j])
        color[i,j,1] = G(img[i][j])

cv2.imshow("color",color)
cv2.waitKey(10000)







