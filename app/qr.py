import cv2 as cv
import numpy as np
import math
import sys
from statistics import mode
from dataclasses import dataclass
from bisect import *
import requests

def kpdst(kp1, kp2):
    """
    Euclidean distance between 2 cv.KeyPoints
    """
    if type(kp1) == cv.KeyPoint:
        dx = kp1.pt[0]-kp2.pt[0]
        dy = kp1.pt[1]-kp2.pt[1]
    else:
        dx = kp1[0]-kp2.pt[0]
        dy = kp1[1]-kp2.pt[1]
      
    return round(math.sqrt(dx*dx + dy*dy), 2)

def update_xrange(low, hi, i, kp, dstmax):
    """
    Updates the indices low and hi, to cover the range
    of keypoints in kp that have X coordinate within dstmax of
    the keypoint at index i. The low index is inclusive, the
    hi index is exclusive, ie one beyond the last point in range.
    """
    while(kp[i].pt[0] - kp[low].pt[0] > dstmax):
        low += 1
    while(hi < len(kp) and kp[hi].pt[0] - kp[i].pt[0] < dstmax):
        hi += 1
    return low, hi

@dataclass
class Padpoint:
    """
    Annotated points that attempt to represent the
    Deluge button pad grid
    """
    i: int                   # index into full list of Padpoints
    kp: object               # cv.KeyPoint found from the image 
    neighbours: list[int]    # List of nearest neighbours
    dists: list[float]       # Distances to neighbours
    oriented: bool = False   # If this padpoint has been oriented to the
                             # suspected grid. If so it's first 4 neighbours
                             # are now ordered +X, +Y, -X, -Y
    u: int = None            # u coordinate on suspected grid
    v: int = None            # v coordinate on suspected grid
    group: int = None        # Contiguous group of points this padpoint was
                             # discovered in.

    def ixy(self):
        """
        Get integer coords for easy drawing.
        """
        return (int(self.kp.pt[0]), int(self.kp.pt[1]))
    


def find_blobs(im, bwmin, bwmax, binary=False):
    """
    Finds black blobs in the thresholded image, to detect the
    coordinates of the button pads.
    """
    params = cv.SimpleBlobDetector_Params()
    # Change thresholds

    if binary:
        params.minThreshold = 100
        params.maxThreshold = 110
        params.thresholdStep = 8
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 12
        params.maxArea = bwmax*bwmax
    else:
        params.minThreshold = 100;
        params.maxThreshold = 255;
        params.thresholdStep = 5; 
        # Filter by Area.
        params.filterByArea = True
        params.minArea = bwmin*bwmin
        params.maxArea = bwmax*bwmax

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.7

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.4

    # Create a detector with the parameters
    ver = (cv.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv.SimpleBlobDetector(params)
    else : 
        detector = cv.SimpleBlobDetector_create(params)

    keypoints = detector.detect(im)
    return keypoints

def threshold_erode(im):
    """
    Adaptive thresholding and erosion/dilation to find thresholded
    blobs that identify the pad grid.
    """
    kernel3 = np.ones((3,3),np.uint8)
    kernel7 = np.ones((7,7),np.uint8)
    #r, im2 = cv.threshold(im, thr, 255, cv.THRESH_BINARY)

    im2 = cv.GaussianBlur(im,(7,7),0)
    im2 = cv.adaptiveThreshold(im2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 59, 0)
    im2 = cv.medianBlur(im2,3)
    
    #cv.imshow("Dotsmax_at", im2)
   
    im2 = cv.dilate(im2, kernel3, 1)
    #cv.imshow("Dotsmax_d", im2)
    im2 = cv.erode(im2, kernel3, 1)
    #cv.imshow("Dotsmax_de", im2)
    im2 = cv.dilate(im2, kernel7, 1)
    #cv.imshow("Dotsmax_ded", im2)
    im2 = cv.erode(im2, kernel3, 1)
    #cv.imshow("Dotsmax_de", im2)
    im2 = cv.dilate(im2, kernel3, 1)

    return im2

def build_grid(umin, umax, vmin, vmax, xoff=0.0, yoff=0.0):
    """
    Builds a numpy array representing an integer-valued
    grid. 
    """
    nx, ny = (umax-umin+1, vmax-vmin+1)
    x = np.linspace(umin+xoff, umax+xoff, nx)
    y = np.linspace(vmin+yoff, vmax+yoff, ny)
    mg = np.array(np.meshgrid(x,y)).T
    return mg

def nearby_points(skp, maxd):
    """
    Scans all keypoints in skp and yields a list
    of the indices of any neighbours within distance
    maxd, and their distances.
    """
    low = 0
    hi = 0
    n = len(skp)
    for i in range(n):
        low, hi = update_xrange(low, hi, i, skp, maxd)
        d = 1e9
        jlist = []
        dlist = []
        for j in range(low, min(hi, n)):
            if i==j:
                continue
            if abs(skp[i].pt[1] - skp[j].pt[1]) < maxd:
                jlist.append(j)
                dlist.append(kpdst(skp[i], skp[j]))
        yield (i, jlist, dlist)
        
def nearest_keypoint(coord, skp, maxd):
    """
    Finds the nearest keypoint in skp to the coordinate coord,
    up to maximum distance maxd.
    """
    low = bisect_right(skp, coord[0]-maxd, key=lambda k:k.pt[0])
    hi = bisect_left(skp, coord[0]+maxd, key=lambda k:k.pt[0])
    dmin = 10000

    nrkp = None
    nj = None
    d = None
    for j in range(low, hi):
        if abs(coord[1] - skp[j].pt[1]) < maxd:
            d = kpdst(coord, skp[j])
            if d < dmin:
                dmin = d
                nrkp = skp[j]
                nj = j
    return dmin, nj, nrkp

def match_grid(grid, skp, maxd):
    """
    Matches a whole grid of xy-points against the image
    keypoints and counts those that line up
    grid - 
    """
    count = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            d, idx, kp = nearest_keypoint(grid[i,j,:], skp, maxd)
            if d < maxd / 5.0:
                count += 1
    return count
            
def label_grid_points(pads, imc):
    """
    Searches all suspected pads, and identify those that look 'gridlike'.
    """
    gridpads = []

    #grid_centroid = None
    ng = 0
    for kp in pads:
        if len(kp.neighbours) == 0:
            continue
        zx = list(zip(kp.neighbours, kp.dists))
        zx.sort(key=lambda x:x[1])
        kp.neighbours, kp.dists = tuple(zip(*zx))

        if point_is_gridlike(pads, kp):
            c = (0, 93, 255)
            ng = ng + 1
            gridpads.append(kp)
        else:
            c = (255, 192, 0)
        
        x,y = kp.ixy()
        if len(zx)>= 4:
            for i in range(4):
                x2,y2 = pads[zx[i][0]].ixy()  
                cv.line(imc, (x,y), (x2, y2), c, thickness=2, lineType=cv.LINE_AA)
    return gridpads


def get_cross_vectors(pads, pp):
    """
    For the 4 closest neighbours to padpoint pp, return
    a list of 4 unit vectors pointing at them from pp.
    """
    x,y = pp.ixy()
    dvs = []
    for i in range(4):
        x2,y2 = pads[pp.neighbours[i]].ixy()
        dv = [x2-x,y2-y]
        dv = [d / math.sqrt(np.dot(dv,dv)) for d in dv]
        dvs.append(dv)
    return dvs

def point_is_gridlike(pads, kp):
    """
    Determines if a padpoint look 'gridlike'. It's 4 closes
    neighbours should form vectors that form a cross. Build the
    vectors and label it gridlike if the 1st vector has two others
    at right angles to it and another pointing in the opposing direction.
    """
    if len(kp.neighbours) < 4:
        return False
    dvs = get_cross_vectors(pads, kp)

    dts = []
    for i in range(3):
        dts.append(np.dot(dvs[i], dvs[3]))
    dts.sort()
    tol = 0.2
    if abs(-1.0 - dts[0]) < tol and abs(dts[1])< tol and abs(dts[2])<tol:
        return True
    else:
        return False

def orient_to_grid(pads, kp, tvect):
    """
    Orients a padpoint to the basis given in tvect.
    This sorts the 1st 4 neighbours into the order
    +U, +V, -U, -V
    """
    dvs = get_cross_vectors(pads, kp)

    dis = [-1000]*4
    for i in range(4):
        gdir = np.dot(dvs[i], tvect)
        if abs(gdir[0]) > abs(gdir[1]):
            # U
            if gdir[0] > 0.0:
                dis[0] = i
            else:
                dis[2] = i
        else:
            # V
            if gdir[1] > 0.0:
                dis[1] = i
            else:
                dis[3] = i
    if sum(dis) == 6:
        kp.neighbours = [kp.neighbours[j] for j in dis]
        pads[kp.i].oriented = True
    else:
        print(kp, dvs)
        #raise Exception("Grid like point failed to orient")
    

def flood_fill_uv_grid(pads, gridpads):
    """
    Searches the gridlike pads, and find all connected groups of
    gridlike pads. They are labelled with a u and v coordinate in
    their group, and and the id of the largest group is returned
    """
    fmax = 0
    gpmax = -1
    grp = 0
    while True:
        p0 = 0
        nflood = 0
        for gp in gridpads:
            if gp.u is None:
                p0 = gp.i
                break
        else:
            break
        
        pads[p0].u = 0
        pads[p0].v = 0

        g = [p0]
        more = True
        i = 0
        uinc = [1,0,-1,0]
        vinc = [0,1,0,-1]
        while i<len(g):
            kp = pads[g[i]]
            if kp.oriented:
                for j in range(4):
                    k = kp.neighbours[j]
                    kp2 = pads[k]
                    if kp2.u is None and kp2.oriented:
                        nflood = nflood + 1
                        kp2.u = kp.u + uinc[j]
                        kp2.v = kp.v + vinc[j]
                        kp2.group = grp
                        g.append(kp2.i)
            i+=1
        if nflood > fmax:
            fmax = nflood
            gpmax = p0
            grpmax = grp
        grp += 1
    return grpmax

def expand_grid(pads, wide_uvgrid, grppnts, limits, grid, skp, maxd):
    """
    Given an incomplete group of grid pads, tries to widen
    it in the +/- U and V directions to cover the full 18x6
    deluge button grid. Matches with raw keypoints in skp to
    tell which direction gives the most benefit.
    """
    expand = [1,1,-1,-1]
    nmatches = [-1,-1,-1,-1]
    
    while sum(nmatches) != 0:

        snaps = [[]]*4
        for i in range(4):
            # Get grid limits in directions relative to 'up'
            # where up is the growth direction
            U, R, D, L = (limits[(i+j)%4] for j in range(4))
           
            # We don't know whether the grid is horizontal or
            # vertical - grow until one axis gets larger than
            # 8 pads, then stop once we hit 8x16 or 16x8
            height = abs(U-D)
            width = abs(L-R)
            if width >= 8:
                if height >= 7 or width >= 15:
                    nmatches[i] = 0
                    continue
               
            row = U + expand[i]

            # Count how many of the saw openCV keypoints we can
            # snap to in the current direction if we expand the grid.
            nmatches[i] = 0
            for j in range(min(R,L), max(R,L)+1):
                if i%2 == 0:
                    d, idx, ikp = nearest_keypoint(grid[row][j], skp, maxd)
                    kpu = row
                    kpv = j
                else:
                    d, idx, ikp = nearest_keypoint(grid[j][row], skp, maxd)
                    kpu = j
                    kpv = row
                if d < maxd/5.0:
                    nmatches[i] += 1
                    pads[idx].u = kpu + int( wide_uvgrid[0][0][0])
                    pads[idx].v = kpv + int(wide_uvgrid[0][0][1])
                    snaps[i].append(pads[idx])

        # Pick the best direction to expand in
        iex = np.argmax(nmatches)
        if nmatches[iex] > 0:
            limits[iex] += expand[iex]
            grppnts += snaps[iex]
            hmg = homography_from_partial_grid(grppnts)
            invh = np.linalg.pinv(hmg)
            grid = cv.perspectiveTransform(np.array([wide_uvgrid.reshape((-1,2))]), invh).reshape(wide_uvgrid.shape)
            # Re score all directions, since this expansion may
            # change the scores on the sides. 
            nmatches = [-1,-1,-1,-1]
        #return limits, grid, hmg
    return limits, grid, hmg


def draw(label, img, keypoints, strfunc):
    """
    Draws an image with keypoints and text annotation
    """
    img2 = img.copy()
    img2 = (img2 // 2) + 126
    for kp in keypoints:
        x = int(kp.kp.pt[0])
        y = int(kp.kp.pt[1])
        cv.putText(img2, strfunc(kp), (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        cv.imshow(label, img2)

def draw_cross(img, x, y, tv):
    """
    Draws a cross on the image (to show basis vectors from PCA)
    """
    x = int(x)
    y = int(y)
    cv.line(img, (x-int(tv[0][0]*40),y-int(tv[0][1]*40)), (x+int(tv[0][0]*40),y+int(tv[0][1]*40)), (0, 255, 0), thickness=4, lineType=cv.LINE_AA)
    cv.line(img, (x-int(tv[1][0]*40),y-int(tv[1][1]*40)), (x+int(tv[1][0]*40),y+int(tv[1][1]*40)), (0, 255, 0), thickness=4, lineType=cv.LINE_AA)

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
            cv.circle(img, (x,y), size, colour, thickness=thickness, lineType=cv.LINE_AA)
            if coords:
                cv.putText(img, f"{i} {j}", (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255))
                cv.putText(img, f"{int(uvs[i,j,0])} {int(uvs[i,j,1])}", (x,y+15), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0))

def homography_from_partial_grid(padpoints):
    corners = [min(padpoints, key=lambda k:k.u+k.v),
               max(padpoints, key=lambda k:k.u-k.v),
               max(padpoints, key=lambda k:k.u+k.v),
               min(padpoints, key=lambda k:k.u-k.v)]
    px_corners = [[kp.kp.pt[0], kp.kp.pt[1]] for kp in corners]
    grid_corners = [[kp.u, kp.v] for kp in corners]

    # Find the ditortion between this (possibly partial) grid and the image
    hmg, status = cv.findHomography(np.array(px_corners), np.array(grid_corners), 0)
    return hmg

def deluge_qr(imgfilename, dbg=True):
    """
    Reads deluge crash handler patterns from image
    files.
    """
    return deluge_qr_img(cv.imread(imgfilename), dbg=dbg)

def deluge_qr_url(url, dbg=False):
    """
    Grab an image from the given URL and attempt to read a deluge
    crash handler pattern.
    """
    response = requests.get(url)
    image_bytes = response.content
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return deluge_qr_img(image)

def deluge_qr_img(img, dbg=False):
    """
    Identifies the deluge button grid in a photo and reads
    off the crash handler pointers and github commit hash.
    """
    imc = img
    im = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    im = (255-im)

    # Scale down to approx 800 px image width/height
    h, w = im.shape[:2]
    scale = max(h / 800.0, w / 800.0)

    h = h // int(scale)
    w = w // int(scale)
    im = cv.resize(im, (w, h), interpolation= cv.INTER_LINEAR)

    #im = cv.bilateralFilter(im, 7, 21, 7)
    #im = cv.medianBlur(im, 5)

    imc = cv.resize(imc, (w, h), interpolation= cv.INTER_LINEAR)
    imdbg = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    imc_clean = imc.copy()
    
    # Lower and upper bound on deluge button sizes, assuming the deluge takes up most of the frame.
    bwmax = max(w,h) // 20
    bwmin = bwmax // 3.5

    dots = threshold_erode(im)
    keypoints = find_blobs(dots, bwmin, bwmax, binary=True)
    if dbg: cv.imshow("Thresholded Pads", dots)
    
    #keypoints = find_blobs(im, bwmin, bwmax)

    skp = list(keypoints)
    skp.sort(key= lambda kp:kp.pt)

    low = 0
    hi = 0
    n = len(skp)
    maxd = bwmax*1.2

    pads = []
    
    for i, js, ds in nearby_points(skp, maxd):
        pads.append(Padpoint(i, skp[i], js,ds))

    median_size = [kp.size for kp in skp][n//2]
    isz = int(median_size)//3
                
    gridpads = label_grid_points(pads, imc)

    grida = np.array([kp.kp.pt for kp in gridpads])
    gridca = np.cov(grida, y=None, rowvar = 0, bias= 1)

    v, vect = np.linalg.eig(gridca)
    tvect = np.transpose(vect)

    draw_cross(imc, w//2, h//2, tvect)

    for kp in gridpads:
        orient_to_grid(pads, kp, vect)

    grpmax = flood_fill_uv_grid(pads, gridpads)

    us = [kp.u for kp in pads if kp.u is not None]
    vs = [kp.v for kp in pads if kp.v is not None]
    minu = min(us)
    minv = min(vs)

    maxu = max(us)
    maxv = max(vs)

    grppnts = [gp for gp in gridpads if gp.group == grpmax]
    
    # Find the ditortion between this (possibly partial) grid and the image
    hmg = homography_from_partial_grid(grppnts)

    invh = np.linalg.pinv(hmg)

    uwid = (maxu-minu)
    vwid = (maxv-minv)

    missing_u = 15-uwid
    missing_v = 15-vwid

    # Cast a wider grid of u/v (pad grid) and x/y (pixel) coordinates, making
    # it wide enough to hold 18*6 pads, regardless of which direction they lie in
    # and which parts of the grid we have already found.
    wide_uvgrid = build_grid(minu-missing_u,maxu+missing_u,minv-missing_v,maxv+missing_v) 
    grid = cv.perspectiveTransform(np.array([wide_uvgrid.reshape((-1,2))]), invh).reshape(wide_uvgrid.shape)
    
    maxu -= int(wide_uvgrid[0][0][0])
    maxv -= int(wide_uvgrid[0][0][1])
    minu -= int(wide_uvgrid[0][0][0])
    minv -= int(wide_uvgrid[0][0][1])

    # Push the grid out into other parts of the image and snap it onto
    # the keypoints found there
    limits = [maxu, maxv, minu, minv]  
    limits, grid, hmg = expand_grid(pads, wide_uvgrid, grppnts, limits, grid, skp, maxd)


    if dbg:
        draw("U", imdbg, pads, lambda k:str(k.u if k.u is not None else " "))
        draw("V", imdbg, pads, lambda k:str(k.v if k.u is not None else " "))
        draw("I", imdbg, pads, lambda k:str(k.i))
    
    maxu, maxv, minu, minv = tuple(limits)
    uvgrid = wide_uvgrid[minu:maxu+1, minv:maxv+1,:] 
    xygrid = grid[minu:maxu+1, minv:maxv+1,:]

    imblk2 = np.zeros(imc.shape, dtype=np.uint8)
    draw_uvgrid(imblk2, xygrid, uvgrid, coords=True)
    
    if maxu-minu < maxv-minv:
        uvgrid = np.flip(np.transpose(uvgrid, axes=(1,0,2)), axis=2)
        xygrid = np.transpose(xygrid, axes=(1,0,2))

    uvgrid[:,:,0] -= uvgrid[0,0,0]
    uvgrid[:,:,1] -= uvgrid[0,0,1]

    corner_uvs = [(0,0), (15,0), (15,7), (0,7)]
    corner_colours = [(0,92,255),(0,255,192), (192, 200, 0), (255, 92, 0)]
    
    for i in range(4):
        u,v = corner_uvs[i]
        cv.circle(imc, (int(xygrid[u][v][0]) , int(xygrid[u][v][1] )), 15, corner_colours[i], thickness=-1, lineType=cv.LINE_AA)
    
    hmg, status = cv.findHomography(
        np.array([xygrid[0][0],
                  xygrid[15][0],
                  xygrid[15][7],
                  xygrid[0][7]]),
        np.array([uvgrid[0][0],
                  uvgrid[15][0],
                  uvgrid[15][7],
                  uvgrid[0][7]]), 0)
    invh = np.linalg.pinv(hmg)

    sidebar = build_grid(16, 17, 0, 7, xoff = 0.7)
    sbgrid = cv.perspectiveTransform(np.array([sidebar.reshape((-1,2))]), invh).reshape(sidebar.shape)

    sidebar2 = build_grid(-2, -1, 0, 7, xoff = -0.7)
    sbgrid2 = cv.perspectiveTransform(np.array([sidebar2.reshape((-1,2))]), invh).reshape(sidebar2.shape)
    
    nright = match_grid(sbgrid, skp, maxd)
    nleft = match_grid(sbgrid2, skp, maxd)

    if nleft > nright:
        sbgrid = sbgrid2

    if maxu-minu < maxv-minv:
        xygrid = np.flip(xygrid, axis=1)
        sbgrid = np.flip(sbgrid, axis=1)
        
    if nleft > nright:
        xygrid = np.flip(xygrid, axis=(0,1))
        sbgrid = np.flip(sbgrid, axis=(0,1))
    
    imblk = np.zeros(imc.shape, dtype=np.uint8)
    draw_uvgrid(imblk, xygrid, uvgrid, coords=True)
    draw_uvgrid(imblk, sbgrid, uvgrid, coords=True)
  
    #cv.imshow("blk",imblk)
    
    # Re-do the homography to make sure it's right way up
    hmg, status = cv.findHomography(
        np.array([xygrid[0][0],
                  xygrid[15][0],
                  xygrid[15][7],
                  xygrid[0][7]]),
        np.array([uvgrid[0][0],
                  uvgrid[15][0],
                  uvgrid[15][7],
                  uvgrid[0][7]]), 0)
    invh = np.linalg.pinv(hmg)
    
    pads_gray = np.full((8, 18), 0, dtype=np.uint8)
    pads_bgr = np.zeros((8, 18, 3), dtype=np.uint8)

    for i in range(16):
        for j in range(8):
            x = int(xygrid[i][j][0])
            y = int(xygrid[i][j][1])
            im4 = im[y-isz:y+isz, x-isz:x+isz]
            pads_gray[j][i] = int(cv.mean(im4)[0])
            im4 = imc_clean[y-isz:y+isz, x-isz:x+isz]
            col = [int(ii) for ii in cv.mean(im4)[:3]]
            pads_bgr[j,i,:] = col

    for i in range(2):
        for j in range(8):
            x = int(sbgrid[i][j][0])
            y = int(sbgrid[i][j][1])
            im4 = im[y-isz:y+isz, x-isz:x+isz]
            pads_gray[j][i+16] = int(cv.mean(im4)[0])
            im4 = imc_clean[y-isz:y+isz, x-isz:x+isz]
            col = [int(ii) for ii in cv.mean(im4)[:3]]
            pads_bgr[j,i+16,:] = col

    out = [0]*18

    #for gp in grid[0]:
    #    cv.circle(imc, (int(gp[0]), int(gp[1])), 7, (0, 255, 255), thickness=4, lineType=cv.LINE_AA)
    
    pads_v = cv.cvtColor(pads_bgr, cv.COLOR_BGR2HSV)[:,:,2]
    pads_grey = cv.cvtColor(pads_bgr, cv.COLOR_BGR2GRAY)

    vrms = pads_v.std()
    grms = pads_grey.std()

    #pg_big = cv.resize(pads_grey, (18*32, 8*32), interpolation= cv.INTER_NEAREST)
    #cv.imshow(f"PG", pg_big)
    if vrms > grms:
        r, pads_v_threshold = cv.threshold(pads_v, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    else:
        r, pads_v_threshold = cv.threshold(pads_grey, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    for ix in range(18):
        for iy in range(8):
            if pads_v_threshold[iy][ix] > 128:
                out[ix] += (0x1 << iy)
    
    for v in out:
        print(f"{v:08b}")
       
    f = []
    for i in range(4):
        f.append(out[i*4]<<24 | out[i*4+1]<<16 | out[i*4+2]<<8 | out[i*4+3])

    f.append(out[16]<<8 | out[17])
    print()
    for fv in f[:4]:
        print(f"0x{fv:08x}")
    print(f"0x{f[4]:04x}")


    bracketpts = [[-.5,-.5],[.5,-.5],
                  [14.5,-.5],[15.5,-.5],
                  [-.5,7.5],[.5,7.5],
                  [14.5,7.5],[15.5,7.5],
                  [-.5,-.5],[-.5,.5],
                  [15.5,-.5],[15.5,.5],
                  [-.5,7.5],[-.5,6.5],
                  [15.5,7.5],[15.5,6.5]]

    bp = cv.perspectiveTransform(np.array([bracketpts]), invh)[0]
    for i in range(8):
        x = int(bp[i*2][0])
        y = int(bp[i*2][1])
        x2 = int(bp[i*2+1][0])
        y2 = int(bp[i*2+1][1])
        cv.line(imc, (x,y), (x2,y2), (0, 255, 255), thickness=4, lineType=cv.LINE_AA)

    pads_gray_big = cv.resize(pads_gray, (18*32, 8*32), interpolation= cv.INTER_NEAREST)
    pads_bgr_big = cv.resize(pads_bgr, (18*32, 8*32), interpolation= cv.INTER_NEAREST)
    pads_v_big = cv.resize(pads_v, (18*32, 8*32), interpolation= cv.INTER_NEAREST)
    pads_v_thr_big = cv.resize(pads_v_threshold, (18*32, 8*32), interpolation= cv.INTER_NEAREST)

    pads_shifted = np.zeros((8, 19), dtype=np.uint8)
    pads_shifted[0:8,0:16] = pads_v_threshold[0:8,0:16]
    pads_shifted[0:8,17:19] = pads_v_threshold[0:8,16:18]
    
    overlay = cv.warpPerspective(pads_shifted, invh, (w, h), flags= cv.INTER_NEAREST)
    overlay = cv.cvtColor(overlay, cv.COLOR_GRAY2BGR)
    comp = cv.bitwise_or(imc, overlay)
    #pads_v_big = cv.resize(pads_v, (18*32, 8*32), interpolation= cv.INTER_NEAREST)
    
    if dbg:        
        cv.imshow("gr", pads_gray)
        cv.imshow("RGB", pads_bgr)
        cv.imshow("Val", pads_v)

        cv.imshow("Pads gray big", pads_gray_big)
        cv.imshow("Pads bgr big", pads_bgr_big)
        
        cv.imshow("Pads v big", pads_v_big)
        cv.imshow("Pads v thr big", pads_v_thr_big)

        cv.imshow("Grey", imdbg)
        cv.imshow("Keypoints", imc)

        cv.imshow("m", comp)
        cv.imshow("pads_v_big", pads_v_big)

        cv.imshow("Clean", imc_clean)

        cv.waitKey(0)

    return f, comp

if __name__ == "__main__":
    try:
        deluge_qr(sys.argv[1])
    except Exception as e:
        # Pause so we can inspect the debug images.
        cv.waitKey(0)
        raise e
