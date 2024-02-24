img_clean = None
import cv2 as cv
from qrcolours import *

FHS = cv.FONT_HERSHEY_SIMPLEX

def circle(img,xy,sz,c, thickness =2):
    """
    Nicer circle drawing
    """
    x,y = xy
    cv.circle(img, (int(x),int(y)), sz, c, thickness=thickness, lineType=cv.LINE_AA)

def pr(ar):
    """
    Prints an array to 2 decimal places
    """
    print([f"{v:2.2f}" for v in ar])

def enlarge(img, factor):
    w,h = img.shape[:2]
    return cv.resize(img, (h*factor, w*factor), interpolation= cv.INTER_NEAREST)

def draw(label, img, keypoints, strfunc):
    """
    Draws an image with keypoints and text annotation
    """
    img2 = img.copy()
    img2 = (img2 // 2) + 126
    for kp in keypoints:
        x = int(kp.kp.pt[0])
        y = int(kp.kp.pt[1])
        cv.putText(img2, strfunc(kp), (x,y), FHS, 0.5, RED)
        cv.imshow(label, img2)

def draw_cross(img, x, y, tv):
    """
    Draws a cross on the image (to show basis vectors from PCA)
    """
    x = int(x)
    y = int(y)
    cv.line(img, (x-int(tv[0][0]*10),y-int(tv[0][1]*10)), (x+int(tv[0][0]*40),y+int(tv[0][1]*40)), YELLOW, thickness=4, lineType=cv.LINE_AA)
    cv.line(img, (x-int(tv[1][0]*10),y-int(tv[1][1]*10)), (x+int(tv[1][0]*40),y+int(tv[1][1]*40)), GREEN, thickness=4, lineType=cv.LINE_AA)

def hollow_rect(img, x, y, w, h, c, t):
    cv.line(img, (x,y), (x,y+h), c, t)
    cv.line(img, (x,y+h), (x+w,y+h), c, t)
    cv.line(img, (x+w,y+h), (x+w,y), c, 6)
    cv.line(img, (x+w,y), (x,y), c, t)

def text(img, text, xy, c, t = 1, size = 1.0):
    cv.putText(img, text, xy, FHS, size, c, t, cv.LINE_AA)

def dbltext(img, text, xy, c, t = 1, size = 1.0):
    cv.putText(img, text, xy, FHS, size, c, t+3, cv.LINE_AA)
    cv.putText(img, text, xy, FHS, size, WHITE, t, cv.LINE_AA)
    
def draw_uvgrid(img, grid, uvs, coords = False, size = 14, colour = (255, 123, 0), thickness=1):
    """
    Debug display of grids of points.
    img -  image to draw to
    grid - grid of point in x-y (pixel) space
    uvs -  corresponding coords in U/V space
           (aligned with the pad grid, but possibly offset/flipped/rotated)
    coords - boolean falg of whether to also draw coords as text.
    size/colour/thickness - pass through to openCV drawing routines
    """
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            x = int(grid[i][j][0])
            y = int(grid[i][j][1]) 
            circle(img, grid[i][j], size, colour, thickness=thickness)
            if coords:
                cv.putText(img, f"{i} {j}", (x,y), FHS, 0.3, (0,255,255))
                cv.putText(img, f"{int(uvs[i,j,0])} {int(uvs[i,j,1])}", (x,y+15), FHS, 0.3, (0,255,0))
