import math
import numpy


def PSNR(img1, img2):   #计算PSNR值用来做客观评价
    D = numpy.array(img1 - img2, dtype=numpy.int64)
    D[:, :] = D[:, :]**2
    RMSE = D.sum()/img1.size
    psnr = 10*math.log10(float(255.**2)/RMSE)
    return psnr


#初始化
def init(img, _blk_size, _Beta_Kaiser):
    """"initialize and return the image after filtering and weighted array. Also make Kaiser window """
    m_shape = img.shape
    m_img = numpy.matrix(numpy.zeros(m_shape, dtype=float))
    m_wight = numpy.matrix(numpy.zeros(m_shape, dtype=float))
    K = numpy.matrix(numpy.kaiser(_blk_size, _Beta_Kaiser))
    m_Kaiser = numpy.array(K.T * K)            # 开一个凯撒窗口
    return m_img, m_wight, m_Kaiser

#定位当前图形窗口
def Locate_blk(i, j, blk_step, block_Size, width, height):

    if i*blk_step+block_Size < width: ##确保图像窗口没有超出图像范围
        point_x = i*blk_step
    else:
        point_x = width - block_Size

    if j*blk_step+block_Size < height:
        point_y = j*blk_step
    else:
        point_y = height - block_Size

    m_blockPoint = numpy.array((point_x, point_y), dtype=int)  # current peak point

    return m_blockPoint

#图像窗口定义
def Define_SearchWindow(_noisyImg, _BlockPoint, _WindowSize, Blk_Size):
    """定义顶点坐标"""
    point_x = _BlockPoint[0]  # current coordinate
    point_y = _BlockPoint[1]

    # 获得SearchWindow四个顶点的坐标
    LX = point_x+Blk_Size/2-_WindowSize/2     # 左上 x
    LY = point_y+Blk_Size/2-_WindowSize/2     # 坐上 y
    RX = LX+_WindowSize                       # 右下 x
    RY = LY+_WindowSize                       # 右下 y

    # judge whether within
    if LX < 0:   LX = 0
    elif RX > _noisyImg.shape[0]:   LX = _noisyImg.shape[0]-_WindowSize
    if LY < 0:   LY = 0
    elif RY > _noisyImg.shape[0]:   LY = _noisyImg.shape[0]-_WindowSize

    return numpy.array((LX, LY), dtype=int)

def Haar(blocks):
    matrix_haar =numpy.array([[1 / 8, 1 / 8, 1 / 4, 0, 1 / 2, 0, 0, 0],[1 / 8, 1 / 8, 1 / 4, 0, -1 / 2, 0, 0, 0],[1 / 8, 1 / 8, -1 / 4, 0, 0, 1 / 2, 0, 0],
     [1 / 8, 1 / 8, -1 / 4, 0, 0, -1 / 2, 0, 0],[1 / 8, -1 / 8, 0, 1 / 4, 0, 0, 1 / 2, 0],[1 / 8, -1 / 8, 0, 1 / 4, 0, 0, -1 / 2, 0],
     [1 / 8, -1 / 8, 0, -1 / 4, 0, 0, 0, 1 / 2],[1 / 8, -1 / 8, 0, -1 / 4, 0, 0, 0, -1 / 2]])

    temp = matrix_haar*blocks*matrix_haar.T

    return temp


def IHaar (temp):
    imatrix_haar =numpy.array([[1,1,1,1,1,1,1,1], [1,1,1,1,-1,-1,-1,-1],
                    [1,1,-1,-1,0,0,0,0],[0,0,0,0,1,1,-1,-1],
                    [1,-1,0,0,0,0,0,0], [0,0,1,-1,0,0,0,0],
                    [0,0,0,0,1,-1,0,0],[0,0,0,0,0,0,1,-1]])

    ihaar = imatrix_haar.T*temp*imatrix_haar
    return ihaar

def Haar_1D (line):
    line_len = len(line)
    if line_len % 2 == 0:
        results = Haar_1D_even(line)
    else:
        results = Haar_1D_odd(line)
    return results

def Haar_1D_even (line):
    line_len = len(line)
    # when len is even number
    haar_temp = numpy.zeros((int(line_len),1))
    line_len = int(line_len)
    for i in range(int(line_len/2)):
        haar_temp[i] = (line[2*i]+line[2*i+1])/2
        haar_temp[i+int(line_len/2)] = (line[2*i]-line[2*i+1])/2
    return haar_temp

def Haar_1D_odd (line):
    len_temp = len(line)
    line_temp2 = numpy.append(line, line[len_temp-1])
    haar_temp = Haar_1D_even(line_temp2)
    return haar_temp

def iHaar_1D (line):
    len_temp = len(line)
    ihaar_temp = numpy.zeros((int(len_temp),1))
    for i in range (int(len_temp/2)):
        ihaar_temp[2*i] = line[i] + line[i+int(len_temp/2)]
        ihaar_temp[2*i+1] = line[i] - line[i+int(len_temp/2)]
    return  ihaar_temp





