import cv2 as cv
import numpy as np
import math
import sys
from statistics import mode
from dataclasses import dataclass
from bisect import *

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
    the keypoitn at index i. The low index is inclusive, the
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
    valid: bool = True       # 
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
    
def threshold_erode(im, thr):
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
    cv.imshow("Dotsmax_dedd", im2)

    return im2

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

def get_cross_vectors(nkp, pp):
    """
    For the 4 closest neighbours to padpoint pp, return
    a list of 4 unit vectors pointing at them from pp.
    """
    x,y = pp.ixy()
    dvs = []
    for i in range(4):
        x2,y2 = nkp[pp.neighbours[i]].ixy()
        dv = [x2-x,y2-y]
        dv = [d / math.sqrt(np.dot(dv,dv)) for d in dv]
        dvs.append(dv)
    return dvs

def point_is_gridlike(nkp, kp):
    """
    Determines if a padpoint look 'gridlike'. It's 4 closes
    neighbours should form vectors that form a cross. Build the
    vectors and label it gridlike if the 1st vector has two others
    at right angles to it and another pointing in the opposing direction.
    """
    if len(kp.neighbours) < 4:
        return False
    dvs = get_cross_vectors(nkp, kp)

    dts = []
    for i in range(3):
        dts.append(np.dot(dvs[i], dvs[3]))
    dts.sort()
    tol = 0.2
    if abs(-1.0 - dts[0]) < tol and abs(dts[1])< tol and abs(dts[2])<tol:
        #print(dts)
        return True
    else:
        return False

def orient_to_grid(nkp, kp, tvect):
    """
    Orients a padpoint to the basis given in tvect.
    This sorts the 1st 4 neighbours into the order
    +U, +V, -U, -V
    """
    dvs = get_cross_vectors(nkp, kp)

    dis = [-1000]*4
    for i in range(4):
        gdir = np.dot(dvs[i], tvect)
        print(kp.i, gdir, dvs[i])
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
        print(kp.i, dis, kp.neighbours)
        nkp[kp.i].oriented = True
    else:
        raise Exception("Grid like point failed to orient")
    
def reject_silk_screen(im, bwmin, bwmax):
    mblobs = 0
    for t in range(10, 255, 10):
        dots = threshold_erode(im, t)
        blobs = find_blobs(dots, bwmin, bwmax, binary=True)
        if len(blobs) > mblobs:
            mblobs = len(blobs)
            imblobs = dots
            bmax = blobs
        print(t, len(blobs))
        break
    return bmax

def build_grid(umin, umax, vmin,vmax):
    nx, ny = (umax-umin+1, vmax-vmin+1)
    x = np.linspace(umin, umax, nx)
    y = np.linspace(vmin, vmax, ny)
    mg = np.array(np.meshgrid(x,y)).T
    return mg

def nearby_points(skp, maxd):
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
    #print(coord, maxd)

    low = bisect_right(skp, coord[0]-maxd, key=lambda k:k.pt[0])
    hi = bisect_left(skp, coord[0]+maxd, key=lambda k:k.pt[0])
    dmin = 10000

    ret = None
    d = None
    for j in range(low, hi):
        if abs(coord[1] - skp[j].pt[1]) < maxd:
            d = kpdst(coord, skp[j])
            if d < dmin:
                dmin = d
                ret = skp[j]
#    print("------")
#    print(coord, "--->")

    return dmin, ret

            
def label_grid_points(nkp, imc):
    gridpoints = []

    #grid_centroid = None
    ng = 0
    for kp in nkp:
        if len(kp.neighbours) == 0:
            continue
        zx = list(zip(kp.neighbours, kp.dists))
        zx.sort(key=lambda x:x[1])
        kp.neighbours, kp.dists = tuple(zip(*zx))

        if point_is_gridlike(nkp, kp):
            c = (0, 93, 255)
            ng = ng + 1
            #if grid_centroid is None:
            #    grid_centroid = [kp.kp.pt[0], kp.kp.pt[1]]
            #else:
            #    grid_centroid[0] = grid_centroid[0] + kp.kp.pt[0]
            #    grid_centroid[1] = grid_centroid[1] + kp.kp.pt[1]
            gridpoints.append(kp)
        else:
            c = (255, 192, 0)
        
        print(kp)
        x,y = kp.ixy()
        if len(zx)>= 4:
            for i in range(4):
                x2,y2 = nkp[zx[i][0]].ixy()  
                cv.line(imc, (x,y), (x2, y2), c, thickness=2, lineType=cv.LINE_AA)
    return gridpoints

def flood_fill_uv_grid(nkp, gridpoints):
    fmax = 0
    gpmax = -1
    grp = 0
    while True:
        p0 = 0
        nflood = 0
        for gp in gridpoints:
            if gp.u is None:
                p0 = gp.i
                break
        else:
            break
        
        nkp[p0].u = 0
        nkp[p0].v = 0

        g = [p0]
        more = True
        i = 0
        uinc = [1,0,-1,0]
        vinc = [0,1,0,-1]
        while i<len(g):
            kp = nkp[g[i]]
            if kp.oriented:
                for j in range(4):
                    k = kp.neighbours[j]
                    kp2 = nkp[k]
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
    print("GPMAX:",gpmax)

def draw(label, img, keypoints, strfunc):
    img2 = img.copy()
    img2 = (img2 // 2) + 126
    for kp in keypoints:
        x = int(kp.kp.pt[0])
        y = int(kp.kp.pt[1])
        cv.putText(img2, strfunc(kp), (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        cv.imshow(label, img2)

def draw_cross(img, x, y, tv):
    x = int(x)
    y = int(y)
    cv.line(img, (x-int(tv[0][0]*40),y-int(tv[0][1]*40)), (x+int(tv[0][0]*40),y+int(tv[0][1]*40)), (0, 255, 0), thickness=4, lineType=cv.LINE_AA)
    cv.line(img, (x-int(tv[1][0]*40),y-int(tv[1][1]*40)), (x+int(tv[1][0]*40),y+int(tv[1][1]*40)), (0, 255, 0), thickness=4, lineType=cv.LINE_AA)

        
def deluge_qr(imgfilename, dbg=1):
    # Setup SimpleBlobDetector parameters.
    imc = cv.imread(imgfilename)
    # Re-read as greyscale and invert
    im = cv.imread(imgfilename, cv.IMREAD_GRAYSCALE)
    im = (255-im)

    # Scale down by a factor of 5
    h, w = im.shape[:2]
    h = h // 5
    w = w // 5
    im = cv.resize(im, (w, h), interpolation= cv.INTER_LINEAR)
    #im = cv.bilateralFilter(im, 7, 21, 7)
    #im = cv.medianBlur(im, 5)

    imc = cv.resize(imc, (w, h), interpolation= cv.INTER_LINEAR)
    imdbg = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    imc_clean = imc.copy()
    
    # Lower and upper bound on deluge button sizes, assuming the deluge takes up most of the frame.
    bwmax = max(w,h) // 20
    bwmin = bwmax // 3.5

    keypoints = reject_silk_screen(im, bwmin, bwmax)
    
    #keypoints = find_blobs(im, bwmin, bwmax)

    skp = list(keypoints)
    skp.sort(key= lambda kp:kp.pt)

    low = 0
    hi = 0
    n = len(skp)
    maxd = bwmax*1.2

    # Find the median distance between blobs - this should approximate
    # the pitch of the deluge key matrix
    all_dsts = []

    bris = []
    cols = []

    nkp = []
    
    for i, js, ds in nearby_points(skp, maxd):
        nkp.append(Padpoint(i, skp[i], js,ds))
        print (i, js, ds)
        if len(ds) > 0:
            all_dsts.append(min(ds))
        else:
            all_dsts.append(1e9)

    print(nkp[0])
            
    dsts = [d for d in all_dsts if d < 1e9]

    if len(dsts) < 2:
        print("maxd too small to link blobs", maxd)
        sys.exit(1)

    dsts.sort()
    #print(dsts)
    median_dist = dsts[len(dsts)//2]
    median_size = [kp.size for kp in skp][n//2]
    isz = int(median_size)//3

    print(len(nkp))

    for kp in nkp:
        for i in range(len(kp.neighbours)-1,0,-1):
            if kp.dists[i] > median_dist*1.6:
                del kp.dists[i]
                del kp.neighbours[i]
    print(len(nkp))
    for kp in nkp:
        n7 = 0
        for i in kp.neighbours:
            if len(nkp[i].neighbours) >= 7:
                n7 += 1
                if n7 > 2:
                    break
        else:
            kp.valid = False
    print(len(nkp))
    gridpoints = label_grid_points(nkp, imc)
    #grid_centroid[0] /= ng
    #grid_centroid[1] /= ng
    #print("GC", grid_centroid)

    print(gridpoints)
    
    grida = np.array([kp.kp.pt for kp in gridpoints])
    gridca = np.cov(grida, y=None, rowvar = 0, bias= 1)

    v, vect = np.linalg.eig(gridca)
    tvect = np.transpose(vect)

    #draw_cross(imc, grid_centroid[0], grid_centroid[1], tvect)

    for kp in gridpoints:
        orient_to_grid(nkp, kp, tvect)

    grpmax = flood_fill_uv_grid(nkp, gridpoints)

    draw("index2", imdbg, nkp, lambda k:str(k.v if k.u is not None else " "))

    us = [kp.u for kp in nkp if kp.u is not None]
    vs = [kp.v for kp in nkp if kp.v is not None]
    minu = min(us)
    minv = min(vs)

    maxu = max(us)
    maxv = max(vs)

    midv = minv + (maxv-minv)//2
    midu = minu + (maxu-minu)//2
    
    pmid = [kp for kp in nkp if kp.u == midu and kp.v == midv]

    if len(pmid) == 0:
        raise Exception("couldn't find middle")

    print("PMID", pmid)

    grppnts = [gp for gp in gridpoints if gp.group == grpmax]
    corners = [min(grppnts, key=lambda k:k.u+k.v),
               max(grppnts, key=lambda k:k.u-k.v),
               max(grppnts, key=lambda k:k.u+k.v),
               min(grppnts, key=lambda k:k.u-k.v)]

    px_corners = [[kp.kp.pt[0], kp.kp.pt[1]] for kp in corners]
    grid_corners = [[kp.u, kp.v] for kp in corners]
    
    hmg, status = cv.findHomography(np.array(px_corners), np.array(grid_corners), 0)
    invh = np.linalg.pinv(hmg)

    print("CORNERS:", corners)

    uwid = (maxu-minu)
    vwid = (maxv-minv)

    missing_u = 15-uwid
    missing_v = 15-vwid

    mg = build_grid(minu-missing_u,maxu+missing_u,minv-missing_v,maxv+missing_v)
    shape = mg.shape
    print(mg.shape)    
    grid = cv.perspectiveTransform(np.array([mg.reshape((-1,2))]), invh).reshape(shape)

    print(grid.shape)

    nx,ny,uv = grid.shape

    maxu -= int(mg[0][0][0])
    maxv -= int(mg[0][0][1])
    minu -= int(mg[0][0][0])
    minv -= int(mg[0][0][1])
    
    for i in range(nx):
        for j in range(ny):
            cv.circle(imc, (int(grid[i][j][0]), int(grid[i][j][1])), 7, (0, 255, 255), thickness=4, lineType=cv.LINE_AA)
    for i in range(minu, maxu+1):
        for j in range(minv, maxv+1):
            cv.circle(imc, (int(grid[i][j][0]), int(grid[i][j][1])), 7, (0, 92, 255), thickness=4, lineType=cv.LINE_AA)

    print("GRIDSHAPE",grid.shape)

    print(mg[0][0])
    print(grid[0][0])
    
    limits = [maxu, maxv, minu, minv]
    expand = [1,1,-1,-1]
    steps = [-1, -1, 1, 1]
    stop = [False]*4
    good = [-1,-1,-1,-1]
    
    while not all(stop):
        expansion = [0]*4

        print("LIMITS:", limits)
        if limits[0]-limits[2] > 7:
            if limits[1]-limits[3] > 6:
                stop[1] = True
                stop[3] = True
            if limits[0]-limits[2] > 14:
                stop[0] = True
                stop[2] = True

        if limits[1]-limits[3] > 7:
            if limits[0]-limits[2] > 6:
                stop[0] = True
                stop[2] = True
            if limits[1]-limits[3] > 14:
                stop[1] = True
                stop[3] = True

        print("STOP:", stop)
                
        i = 0
        while min(good) < 0:
            print(good, stop)
            if good[i] > 0:
                i+=1
                continue
            if stop[i]:
                good[i] = 0
                i+=1
                continue
            l1 = limits[(i+1)%4]
            l2 = limits[(i+3)%4]
            start = min(l1,l2)
            end = max(l1,l2)
            cc2 = limits[i] + expand[i]

            ngood = 0
            for j in range(start, end+1):
                if i%2 == 0:
                    d, ikp = nearest_keypoint(grid[cc2][j], skp, maxd)
                else:
                    d, ikp = nearest_keypoint(grid[j][cc2], skp, maxd)
                if d < maxd/5.0:
                    ngood = ngood + 1
            good[i] = ngood
            print(l1,l2,start,end,cc2,ngood)
            i += 1
            
        iex = np.argmax(good)
        if not stop[iex]:
            print("GOOD:",good)
            print("IEX:", iex)

            limits[iex] += expand[iex]
        good[iex] = -1
    print(limits)

    maxu, maxv, minu, minv = tuple(limits)
    for i in range(minu, maxu+1):
        for j in range(minv, maxv+1):
            cv.circle(imc, (int(grid[i][j][0]), int(grid[i][j][1])),
                      7, (255, 123, 0), thickness=4, lineType=cv.LINE_AA)
    
    # Get colour / brightness values from
    # each keypoint by averaging out a small rectangle
    bris = []
    for kp in skp:
        x,y = kp.pt
        x = int(x)
        y = int(y)
        im4 = im[y-isz:y+isz, x-isz:x+isz]
        bri = int(cv.mean(im4)[0])
        bris.append(bri)
        im4 = imc_clean[y-isz:y+isz, x-isz:x+isz]
        col = [int(ii) for ii in cv.mean(im4)[:3]]
        cols.append(col)

    kpa2 = []
    maxd = median_dist*1.6
    low = 0
    hi = 0

    #print(median_dist)
    cv.circle(imdbg, (int(skp[0].pt[0]), int(skp[0].pt[1])), int(maxd), (0, 0, 255), thickness=2, lineType=cv.LINE_AA)

    if dbg == 1:
        cv.circle(imdbg, (w//2,h//2), int(bwmin), (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
        cv.circle(imdbg, (w//2,h//2), int(bwmax), (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
 
    filtered_keypoints = [nk for nk in nkp if len(nk.neighbours) >= 7 and nk.valid ]

    imc = cv.drawKeypoints(imc, skp, np.array([]), (255,255,0), 0)
    for i in range(n):
        cv.circle(imc, (int(skp[i].pt[0]), int(skp[i].pt[1])), isz, (bris[i], bris[i], bris[i]), thickness=2, lineType=cv.LINE_AA)

    draw("nngh", imdbg, filtered_keypoints, lambda k:str(len(k.neighbours)))
    draw("index", imdbg, nkp, lambda k:str(k.i))
    
    old_filtered_keypoints = [nk.kp for nk in nkp if len(nk.neighbours) >= 7 ]
    imc = cv.drawKeypoints(imc, old_filtered_keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    if len(filtered_keypoints) < 4:
        print("Too few blobs remained after filtering")
        # Show keypoints
        if dbg == 1:
            cv.imshow("Grey", imdbg)
            cv.imshow("Keypoints", imc)
            cv.waitKey(0)
        sys.exit(1)

    a_all = np.array([kp.pt for kp in skp])
    
    a = np.array([kp.kp.pt for kp in filtered_keypoints])
    ca = np.cov(a, y=None, rowvar = 0, bias= 1)

    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)

    ar = np.dot(a,np.linalg.inv(tvect))

    print(vect)
    print(tvect)
    
    x = w//2
    y = h//2
    cv.line(imc, (x-int(tvect[0][0]*40),y-int(tvect[0][1]*40)), (x+int(tvect[0][0]*40),y+int(tvect[0][1]*40)), (255, 192, 0), thickness=4, lineType=cv.LINE_AA)
    cv.line(imc, (x-int(tvect[1][0]*40),y-int(tvect[1][1]*40)), (x+int(tvect[1][0]*40),y+int(tvect[1][1]*40)), (255, 192, 0), thickness=4, lineType=cv.LINE_AA)
    
    mina = np.min(ar,axis=0)
    maxa = np.max(ar,axis=0)

    c0 = a[:,0] + a[:,1]
    c1 = a[:,0] - a[:,1]
    corners = []
    corners.append(a[np.argmin(c0),:])
    corners.append(a[np.argmax(c1),:])
    corners.append(a[np.argmin(c1),:])
    corners.append(a[np.argmax(c0),:])

    print(corners)
    
    cc = [(0,92,255),(0,255,192), (192, 200, 0), (255, 92, 0)]

    if abs(corners[0][0] - corners[1][0]) < abs(corners[0][1] - corners[2][1]):
        corners = [corners[i] for i in [2,0,3,1]]

    # Draw detected blobs as red circles.

    hmg, status = cv.findHomography(np.array(corners), np.array([[1.0,1.0],[14.0,1.0],[1.0,6.0],[14.0,6.0]]), 0)
    invh = np.linalg.pinv(hmg)

    found = cv.perspectiveTransform(np.array([a]), hmg)

    nx, ny = (16, 8)
    x = np.linspace(0, 15, nx)
    y = np.linspace(0, 7, ny)
    mg = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)

    nx, ny = (2, 8)
    x2 = np.linspace(16.7, 17.7, nx)
    y2 = np.linspace(0, 7, ny)
    
    mg1 = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)
    mg2 = np.array(np.meshgrid(x2,y2)).T.reshape(-1, 2)

    mg = np.concatenate((mg1,mg2), 0)
    
    grid = cv.perspectiveTransform(np.array([mg]), invh)

    #for gp in grid[0]:
    #    cv.circle(imc, (int(gp[0]), int(gp[1])), 7, (0, 255, 255), thickness=4, lineType=cv.LINE_AA)

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


    skpgrid = cv.perspectiveTransform(np.array([a_all]), hmg)[0]

    out = [0]*18

    midbri = (max(bris)-min(bris)) // 2

    gray_image = np.full((8, 20), 0, dtype=np.uint8)

    sat_image = np.zeros((8, 20, 3), dtype=np.uint8)

    nleft = 0
    nright = 0
    
    for i in range(n):
        gx, gy = skpgrid[i]
        if gx > 16.0:
            gx -= 0.7
        if gx < -1.0:
            gx += 0.7
        ix = round(gx)
        iy = round(gy)
        #cv.putText(imc, str(i), (int(skp[i].pt[0]), int(skp[i].pt[1])), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255))
        #print(i, gx, gy, ix,iy,bris[i])
        if ix>=-2 and ix<=17 and iy >=0 and iy <=7:
            gray_image[iy][ix+2] = bris[i]
            sat_image[iy][ix+2][:] = cols[i]
            if ix < 0:
                nleft+=1
            if ix > 15:
                nright+=1

    if nleft > nright:
        gray_image = gray_image[7::-1,17::-1]
        sat_image = sat_image[7::-1,17::-1,:]
        corners = [corners[i] for i in [3,2,1,0]]
        hmg, status = cv.findHomography(np.array(corners), np.array([[1.0,1.0],[14.0,1.0],[1.0,6.0],[14.0,6.0]]), 0)
        invh = np.linalg.pinv(hmg)

    else:
        gray_image = gray_image[:,2:]
        sat_image = sat_image[:,2:,:]

    i = 0
    for c in corners:
        cv.circle(imc, (int(c[0]), int(c[1])), 5, cc[i], thickness=4, lineType=cv.LINE_AA)
        i+=1
        
    sat_img2 = cv.cvtColor(sat_image, cv.COLOR_BGR2HSV)[:,:,2]
    if dbg==1:
        cv.imshow("Satonly", sat_img2)

    sat_img2 = cv.adaptiveThreshold(sat_img2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 0)

    
    
    #r, sat_img2 = cv.threshold(sat_img2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 0)

    for ix in range(18):
        for iy in range(8):
            if sat_img2[iy][ix] > 128:
                out[ix] += (0x1 << iy)


    hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])

    i = 0
#    for v in hist:
#        print( i, "*"*int(v))
#        i += 1

    for v in out:
        print(f"{v:08b}")

       
    # Show keypoints
    if dbg == 1:
        cv.imshow("Grey", imdbg)
        cv.imshow("Keypoints", imc)

    gray_image = cv.resize(gray_image, (18*32, 8*32), interpolation= cv.INTER_NEAREST)
    if dbg == 1: cv.imshow("Grey2", gray_image)
    sat_image = cv.resize(sat_image, (18*32, 8*32), interpolation= cv.INTER_NEAREST)
    if dbg == 1: cv.imshow("Sat", sat_image)


    f = []
    for i in range(4):
        f.append(out[i*4]<<24 | out[i*4+1]<<16 | out[i*4+2]<<8 | out[i*4+3])

    f.append(out[16]<<8 | out[17])
    print()
    for fv in f[:4]:
        print(f"0x{fv:08x}")
    print(f"0x{f[4]:04x}")

    overlay = cv.warpPerspective(sat_img2, invh, (w, h), flags= cv.INTER_NEAREST)

    gc = [[-.5,-.5],[-.5,7.5],
          [15.5,7.5],[15.5,-.5]]

    corn = cv.perspectiveTransform(np.array([gc]), invh)[0]
    corn = np.rint(corn).astype(int)

    mask = np.full((h,w), 255, dtype=np.uint8)

    cv.fillPoly(mask, [corn], 0)

    m3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    #comp = cv.bitwise_and(imc, m3)
    overlay = cv.cvtColor(overlay, cv.COLOR_GRAY2BGR)
    comp = cv.bitwise_or(imc, overlay)

    if dbg == 1: cv.imshow("m", comp)

    sat_img2 = cv.resize(sat_img2, (18*32, 8*32), interpolation= cv.INTER_NEAREST)
    if dbg == 1: cv.imshow("Satonly", sat_img2)

    if dbg == 1: cv.waitKey(0)

    return f

if __name__ == "__main__":
    deluge_qr(sys.argv[1])
