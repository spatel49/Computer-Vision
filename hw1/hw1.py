import numpy as np
import random
import cv2
import math
import itertools

def filtermatrix(filter_matrix, img):
    height, width = img.shape[:2]
    result = np.zeros((height, width), np.uint8)
    for i in range(1, height-1):
        for j in range(1, width-1):
            fval = filterh(i, j, filter_matrix, img)
            result[i][j] = fval
    return result


def filterh(x, y, filter_matrix, img):
    sum = 0
    for i in range(3):
        for j in range(3):
            sum += img[x + i - 1][y + j - 1] * filter_matrix[i][j]
    if (sum <= 0):
        return 0
    elif (sum >= 255):
        return 255
    return sum

def nonmaxsuppress(img, Mx, My):
    height, width = img.shape[:2]
    supression = np.power(np.power(Mx, 2.0) + np.power(My, 2.0), 0.5)
    theta = np.arctan2(Mx, My)
    cv2.imwrite('supression.png', supression)
    for i in range(height):
        for j in range(width):
            if (i == 0 or i == height-1 or j == 0 or j == width - 1):
                Mx[i][j] = My[i][j] = 0
                continue
            degree = theta[i][j] * 180.0/np.pi
            if (degree < 0):
                degree += 180
            if (((0 <= degree < 22.5) or (157.5 <= degree <= 180)) and (supression[i][j] <= supression[i][j-1] or supression[i][j] <= supression[i][j+1]) and (supression[i][j] <= supression[i-1][j+1] or supression[i][j] <= supression[i+1][j-1]) and (supression[i][j] <= supression[i-1][j] or supression[i][j] <= supression[i+1][j]) and (supression[i][j] <= supression[i-1][j-1] or supression[i][j] <= supression[i+1][j+1])):
                Mx[i][j] = My[i][j] = 0
    return [Mx, My]

def threshold(img):
    height, width = img.shape[:2]
    result = np.zeros((height, width), dtype=np.int32)
    strong_i, strong_j = np.where(img >= (img.max() * 0.20))
    zeros_i, zeros_j = np.where(img < (img.max() * 0.20 * 0.02))
    weak_i, weak_j = np.where((img <= (img.max() * 0.20)) & (img >= (img.max() * 0.20 * 0.02)))
    result[strong_i, strong_j] = np.int32(255)
    result[weak_i, weak_j] = 0
    return result

def hessian(img, Ixy, Iyy, Ixx):
    height,width = img.shape[:2]
    result = img.copy()*0

    for i in range(2, height-2):
        for j in range(2, width-2):
            determinant = np.sum(Ixx[i-2:i+1+2, j-2:j+1+2]) * np.sum(Iyy[i-2:i+1+2, j-2:j+1+2]) - np.sum(Ixy[i-2:i+1+2, j-2:j+1+2])**2
            trace = np.sum(Ixx[i-2:i+1+2, j-2:j+1+2]) + np.sum(Iyy[i-2:i+1+2, j-2:j+1+2])
            r = determinant - .06*(trace**2)
            if r > 300000:
                result[i][j] = r

    for i in range(1, height-1):
        for j in range(1, width-1):
            if result[i][j] == 0:
                return result
            height, width = result.shape
            if j <= 1 or j >= width - 1 - 1 or i <= 1 or i >= height - 1 - 1:
                return result
            for ix in range(-1, 1+1):
                for jy in range(-1, 1+1):
                    if result[i][j] < result[i+ix][j+jy]:
                        result[i][j] = 0
                        return result
                    else:
                        result[i+ix][j+jy] = 0
            result[i][j] = 255

    for i in range(1, height-1):
        for j in range(1, width-1):
            result[i][j] = result[i][j] if result[i][j] > img[i][j] else img[i][j]

    return result

def distance(point1, point2, point3):
    if point1 == point2:
        return np.sqrt((point2[0]-point3[0])**2 + (point2[1] - point3[1])**2)
    return np.linalg.norm(np.cross(np.array(point2)-np.array(point1),np.array(point3)-np.array(point1)))/np.linalg.norm(np.array(point2)-np.array(point1))
    
def ransac(img):
    height, width = img.shape[:2]
    result = img.copy()
    pix = []

    for i in range(1, height-1):
        for j in range(1, width-1):
            if result[i][j] == 255:
                pix += [[i,j]]


    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    count = 0
    colors = [[0, 0, 255], 
            [255, 255, 0], 
            [0, 255, 0], 
            [0, 255, 255]]
    counted = []

    while count < 4:
        arr = []
        start = pix[np.random.randint(0, len(pix)-1)]
        end = pix[np.random.randint(0, len(pix)-1)]
        if counted:
            for pixel in counted:
                while distance(start, end, pixel) < 15:
                    start = pix[np.random.randint(0, len(pix)-1)]
                    end = pix[np.random.randint(0, len(pix)-1)]
        arr += [start, end]
        pixclose = 2

        for pixel in pix:
            dist = distance(start, end, pixel)
            if dist < 10:
                arr += [pixel]
                pixclose += 1

        if pixclose >= 10:
            for pixel in arr:
                result[pixel[0]][pixel[1]] = colors[count]
                result[pixel[0]-1][pixel[1]-1] = colors[count]
                result[pixel[0]-1][pixel[1]] = colors[count]
                result[pixel[0]-1][pixel[1]+1] = colors[count]
                result[pixel[0]][pixel[1]-1] = colors[count]
                result[pixel[0]][pixel[1]+1]= colors[count]
                result[pixel[0]+1][pixel[1]-1] = colors[count]
                result[pixel[0]+1][pixel[1]] = colors[count]
                result[pixel[0]+1][pixel[1]+1] = colors[count]
                if pixel == start or pixel == end:
                    continue
                pix.remove(pixel)
            finaldis = distance(start, start, end)
            finalp = [start, end]
            for i in range(len(arr)):
                for j in range(len(arr)):
                    dist = distance(arr[i], arr[i], arr[j])
                    if dist > finaldis:
                        finaldis = dist
                        finalp = [arr[i], arr[j]]
            result = cv2.line(result, tuple([finalp[0][1], finalp[0][0]]), tuple([finalp[1][1], finalp[1][0]]), colors[count], 1)
            count += 1
            counted += finalp

    for pixel in counted:
        result[pixel[0]][pixel[1]] = [255, 0, 0]
        for i in range(-2,3):
            for j in range(-2,3):
                result[pixel[0]+i][pixel[1]+j] = [255,0,0]
    return result

if __name__ == "__main__":
    gauss = [[0.077847, 0.123317, 0.077847],[0.123317, 0.195346, 0.123317],[0.077847, 0.123317, 0.077847]]
    sobelx = [[1, 2, 1],[0, 0, 0],[-1, -2, -1]]
    sobely = [[1, 0, -1],[2, 0, -2],[1, 0, -1]]

    roadimg = cv2.imread("road.png", 0)
    
    gaussianfiltered = filtermatrix(gauss, roadimg)
    x = filtermatrix(sobelx, gaussianfiltered)
    y = filtermatrix(sobely, gaussianfiltered)
    x,y = nonmaxsuppress(gaussianfiltered, x, y)
    x = threshold(x)
    y = threshold(y)

    cv2.imwrite("gauss.png", gaussianfiltered)
    cv2.imwrite('x.png', x)
    cv2.imwrite('y.png', y)

    result = x.copy()
    height,width = x.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            result[i][j] = 50 if result[i][j] > 0 or y[i][j] > 0 else 0
    cv2.imwrite('composite.png', result)

    Ixy = filtermatrix(sobelx, y)
    Iyy = filtermatrix(sobely, y)
    Ixx = filtermatrix(sobelx, x)

    hessianf = hessian(result, Ixy, Iyy, Ixx)
    cv2.imwrite('corner.png', hessianf)

    hessf = cv2.imread('corner.png', 0)
    rnsc = ransac(hessf)
    cv2.imwrite('ransac.png', rnsc)
    print("Images are now in folder")