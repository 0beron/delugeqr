import cv2 as cv
import numpy as np
import math
import sys
import hough
from statistics import mode
from dataclasses import dataclass, astuple
from bisect import *
import requests
import argparse
import qrdebug
from qrdebug import circle
import github
import os
from numpy.random import default_rng

from qrcolours import *

rng = default_rng()

ERODE_HIGH = 0
ERODE_LOW = 1

BOTH = 0
GRID = 1
HOUGH = 2

GREY = 0
VALUE = 1
LUMA = 2


BC_THRESHOLD = 2e-2

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
        params.minArea = 30
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

def threshold_erode(im, erode_amount, fill_holes):
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
    if erode_amount == ERODE_LOW:
        
        return im2
   
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

def fill_holes(im):
    """
    Fills in holes in the thresholded image, to try and
    repair pads that have become hollow.
    """
    im2inv = cv.bitwise_not(im)
    contour, hier = cv.findContours(im2inv,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
            
    for cnt in contour:
        cv.drawContours(im2inv,[cnt],0,255,-1)

    im2 = cv.bitwise_not(im2inv)
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
            x,y = kp.ixy()
            for ign in kp.neighbours:
                x2,y2 = pads[ign].ixy()  
                cv.line(imc, (x,y), (x2, y2), c, thickness=3, lineType=cv.LINE_AA)
        else:
            c = (255, 192, 0)
        
            x,y = kp.ixy()
            if len(zx)>= 4:
                for i in range(4):
                    x2,y2 = pads[zx[i][0]].ixy()  
                    cv.line(imc, (x,y), (x2, y2), c, thickness=1, lineType=cv.LINE_AA)
    return gridpads

def locate_grid(im, imc, bwmin, bwmax):
    """
    Try various combinations of thresholding / erosion / hole filling
    and return the results that yield the most potential pads.
    """
    results = [attempt_grid(im, imc, bwmin, bwmax, ERODE_HIGH, False),
               attempt_grid(im, imc, bwmin, bwmax, ERODE_HIGH, True), 
        attempt_grid(im, imc, bwmin, bwmax, ERODE_LOW, False), 
        attempt_grid(im, imc, bwmin, bwmax, ERODE_LOW, True)]
    
    num = np.array([len(r[2]) for r in results])
    best = np.argmax(num)

    return results[best]

def attempt_grid(im, imc, bwmin, bwmax, erode_amount, hole_fill):
    """
    Threshold the image looking for pads
      im - black and white image to threshold
      imc - colour image for debug
      bwmin - lower bound on pad size in pixels
      bwmax - upper bound on pad size in pixels
      erode_amount - ERODE_LOW or ERODE_HIGH
      hole_fill - whether to fill holes in the threshold image
    """
    dots = threshold_erode(im, erode_amount, hole_fill)
    if hole_fill:
        dots = fill_holes(dots)
    
    keypoints = find_blobs(dots, bwmin, bwmax, binary=True)
   
    skp = list(keypoints)
    skp.sort(key= lambda kp:kp.pt)
    imc2 = imc.copy()
    low = 0
    hi = 0
    maxd = bwmax*1.2

    pads = []
    
    for i, js, ds in nearby_points(skp, maxd):
        pads.append(Padpoint(i, skp[i], js,ds))
  
    gridpads = label_grid_points(pads, imc2)

    return skp, pads, gridpads, dots, imc2

def get_cross_vectors(pads, pp):
    """
    For the up to 4 closest neighbours to padpoint pp, return
    a list of unit vectors + distances pointing at them from pp.
    """
    x,y = pp.ixy()
    dvs = []
    dsts = []
    for i in range(min(4, len(pp.neighbours))):
        x2,y2 = pads[pp.neighbours[i]].ixy()
        dv = [x2-x,y2-y]
        dst = math.sqrt(np.dot(dv,dv))
        dv = [d / dst for d in dv]
        dvs.append(dv)
        dsts.append(dst)
    return dvs, dsts

def point_is_gridlike(pads, kp):
    """
    Determines if a padpoint looks 'gridlike'. It's 4 closest
    neighbours should form vectors that form a cross. Build the
    vectors and label it gridlike if the 1st vector has two others
    at right angles to it and another pointing in the opposing direction.
    """
    if len(kp.neighbours) < 3:
        return False
    dvs, dsts = get_cross_vectors(pads, kp)
    compare = [-1.0, 0.0, 0.0]
    compare2 = [0.0, 1.0, -1.0]
    tol = 0.2
    sums = []
    pairs=[]
    for i in range(len(dvs)):
        found = [0,0,0]
        pairs.append(set())
        for j in range(len(dvs)):
            if i != j:
                dt = np.dot(dvs[i], dvs[j])
                dvir = [dvs[i][1], -dvs[i][0]]
                dtr = np.dot(dvir, dvs[j])
                for k in range(3):
                    if abs(compare[k] - dt) < tol and abs(compare2[k]-dtr) < tol and found[k] == 0:
                        found[k] = 1
                        pairs[-1].add(i)
                        pairs[-1].add(j)

        sums.append(sum(found))
   
    if max(sums)>=2:
        ibest = np.argmax(sums)
        used = list(pairs[ibest])
        used.sort()
        kp.neighbours = [kp.neighbours[i] for i in used]
        dsts2 = [dsts[i] for i in used]
        if (max(dsts2)/min(dsts2) < 1.2):
            return True

    return False


def orient_to_grid(pads, kp, tvect):
    """
    Orients a padpoint to the basis given in tvect.
    This sorts the 1st 4 neighbours into the order
    +U, +V, -U, -V
    """
    dvs, dsts = get_cross_vectors(pads, kp)

    dis = [-1000]*4
    for i in range(len(dvs)):
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
    elif sum(dis) > -1000:
        ng2 = []
        for j in dis:
            if j >= 0:
                ng2.append(kp.neighbours[j])
            else:
                ng2.append(None)
        kp.neighbours = ng2[:]
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
    grpmax = None
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
                    if k is None:
                        continue
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
    hmg = None
    while sum(nmatches) != 0:

        snaps = [[], [], [], []]
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
            if height >= 15:
                nmatches[i] = 0
                continue
               
            row = U + expand[i]

            # Count how many of the raw openCV keypoints we can
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
                    pads[idx].u = kpu + int(wide_uvgrid[0][0][0])
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
    if hmg is None:
        hmg = homography_from_partial_grid(grppnts)
        invh = np.linalg.pinv(hmg)
        grid = cv.perspectiveTransform(np.array([wide_uvgrid.reshape((-1,2))]), invh).reshape(wide_uvgrid.shape)
    
    return limits, grid, hmg

def trim_partial_grid(convex_pads):
    """
    Finds rows and columns of a partial grid and drops
    those that have only one entry, since these are more
    likely to cause the corner finding process to choose
    this one pad for more than one corner, which makes the
    keystone/homography degenerate.
    """
    altered = False
    
    extents = [min(convex_pads, key=lambda k:k.u).u,
                max(convex_pads, key=lambda k:k.u).u,
                min(convex_pads, key=lambda k:k.v).v,
                max(convex_pads, key=lambda k:k.v).v]
    
    for u in range(extents[0], extents[1]+1):
        row = [pnt for pnt in convex_pads if pnt.u == u]
        if len(row) == 1 and row[0] in convex_pads:
            convex_pads.remove(row[0])
            altered = True
    for v in range(extents[2], extents[3]+1):
        row = [pnt for pnt in convex_pads if pnt.v == v]
        if len(row) == 1 and row[0] in convex_pads:
            convex_pads.remove(row[0])
            altered = True
    return convex_pads, altered


def homography_from_partial_grid(padpoints):
    """
    Finds 4 corners from the list of supplied padpoints, and
    calculates the homography matrix between x/y (pixel) space
    and u/v (pad grid space). The list of padpoints must have
    u and v coordinates set.
    """
    more = True
    convex_pads = padpoints[:]

    while more:
        more = False

        # Find corners of the the available points - for a neat rectangle
        # this sweeps the u+v = constant and u-v = constant lines to get the corners
        # For a diamond of points it risks finding the same corner for a true diagonal
        # sweep so we tilt the sweep lines by a tiny amount to favour clockwise corners     
        corners = [min(convex_pads, key=lambda k:1.05*k.u+0.95*k.v),
                max(convex_pads, key=lambda k:0.95*k.u-1.05*k.v),
                max(convex_pads, key=lambda k:1.05*k.u+0.95*k.v),
                min(convex_pads, key=lambda k:0.95*k.u-1.05*k.v)]
        
        uvcorners = [(c.u , c.v) for c in corners]

        unique_corners = set(uvcorners)
        if len(unique_corners) < 4:
            convex_pads, altered = trim_partial_grid(convex_pads)

            if not altered:
                raise Exception("Failed to find homography from partial grid")
            else:
                more = True
                continue

        px_corners = [[kp.kp.pt[0], kp.kp.pt[1]] for kp in corners]
        grid_corners = [[kp.u, kp.v] for kp in corners]
        # Find the ditortion between this (possibly partial) grid and the image
        hmg, status = cv.findHomography(np.array(px_corners), np.array(grid_corners), 0)

    return hmg

def extents(grppnts):
    us = [kp.u for kp in grppnts if kp.u is not None]
    vs = [kp.v for kp in grppnts if kp.v is not None]
    minu = min(us)
    minv = min(vs)
    maxu = max(us)
    maxv = max(vs)
    return minu, minv, maxu, maxv

def bhattacharyya_coefficient(mu1, sigma1, mu2, sigma2):
    term1 = 0.25 * ((mu1 - mu2) ** 2) / (sigma1 ** 2 + sigma2 ** 2)
    term2 = 0.5 * np.sqrt((sigma1 ** 2) / (sigma1 ** 2 + sigma2 ** 2) + (sigma2 ** 2) / (sigma1 ** 2 + sigma2 ** 2))
    bc = np.exp(-term1) * term2
    return bc

def threshold_grid_section(name, img_sect, dbg):
    # Try 3 ways of telling lit from unlit
    # - overall grey value
    # - Value from HSV
    # - Luma from LAB
    greys = [cv.cvtColor(img_sect, cv.COLOR_BGR2GRAY),
             cv.cvtColor(img_sect, cv.COLOR_BGR2HSV)[:,:,2],
             cv.cvtColor(img_sect, cv.COLOR_BGR2LAB)[:,:,0]]

    # Threshold using the OTSU 2-peak histogram method
    thr = [cv.threshold(g, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1] for g in greys]

    # Find the standard deviation of the colours in the lit pixels
    # A small value means the pixels are very similar in grey value, if
    # the thresholding pulled in an incorrect pixel it will increase this value, eg
    # if the lit pixels are all white (255), and the unlit pixels are widely spread and
    # include some lighter values, one may be light enough for OTSU to group it
    # with the lit pixels but this will greatly increase the std dev of the lit pixels
    mean_on, std_on = zip(*[cv.meanStdDev(gt[0], mask=gt[1]) for gt in zip(greys,thr)])
    mean_off, std_off = zip(*[cv.meanStdDev(gt[0], mask=cv.bitwise_not(gt[1])) for gt in zip(greys,thr)])
  
    h, w = img_sect.shape[:2]
    h *= 32
    w *= 32

    # Get a measure of how much the histogram curves for the lit pads
    # overlaps with the unlit ones. Too much overlap suggests actually there
    # are no lit pads at all.

    std_on_copy = list(std_on)
    bcs = [bhattacharyya_coefficient(mean_on[i], std_on[i], mean_off[i], std_off[i]) for i in range(3)]
    for i in range(3):
        if bcs[i] > BC_THRESHOLD:
            # Suspected section of grid with no pads lit
            # Set the thresholded image to all 0
            thr[i] *= 0
            std_on_copy[i] = 255
    
    # Pick whichever gives us the smallest std dev for lit pixels.
    # But ignore options that have too much overlap from the test above
    ind = std_on_copy.index(min(std_on_copy))
    
    if dbg:
        # Plot all the above for debugging
        spacer = np.full((h, 16, 3), (255,0,0), dtype=np.uint8)
        big_bgr = cv.resize(img_sect, (w, h), interpolation= cv.INTER_NEAREST)
        bigs = [big_bgr]
        row2 = [big_bgr]
        for i in range(3):

            gbig = cv.cvtColor(cv.resize(greys[i], (w, h), interpolation= cv.INTER_NEAREST), cv.COLOR_GRAY2BGR)
            tbig = cv.cvtColor(cv.resize(thr[i], (w, h), interpolation= cv.INTER_NEAREST), cv.COLOR_GRAY2BGR)

            qrdebug.dbltext(gbig, ["GREY", "VALUE", "LUMA"][i], (10,30), BC, 1, size=0.5)
            qrdebug.dbltext(gbig, "std dev on", (10,45), BC, 1, size=0.5)
            qrdebug.dbltext(gbig, f"{std_on[i][0,0]:.2f}", (10,75), BC, 1)
            qrdebug.dbltext(gbig, "std dev off", (10,90), BC, 1, size=0.5)
            qrdebug.dbltext(gbig, f"{std_off[i][0,0]:.2f}", (10,120), BC, 1)
            qrdebug.dbltext(gbig, "mean on", (10,135), BC, 1, size=0.5)
            qrdebug.dbltext(gbig, f"{mean_on[i][0,0]:.2f}", (10,165), BC, 1)
            qrdebug.dbltext(gbig, "mean off", (10,180), BC, 1, size=0.5)
            qrdebug.dbltext(gbig, f"{mean_off[i][0,0]:.2f}", (10,210), BC, 1)
            qrdebug.dbltext(gbig, f"BC: {bcs[i][0,0]}", (10,240), BC, 1, size=0.5)
            #print(f"--mean_on:{mean_on[i]}")
            #print(f"  mean_off:{mean_off[i]}")
            #print(f"  std_on:{std_on[i]}")
            #print(f"  std_off:{std_off[i]}")
            #print(abs(mean_on[i] - mean_off[i]))
            #print(3.0*(max(std_off[i], std_on[i])))
            #print(bcs[i])
        
            if i == ind:
                qrdebug.hollow_rect(tbig, 0,0,w,h, GREEN, 6)
            if bcs[i] > BC_THRESHOLD:
                qrdebug.hollow_rect(tbig, 0,0,w,h, RED, 3)

            bigs.append(spacer)
            bigs.append(gbig)
            row2.append(spacer)
            row2.append(tbig)

        cv.imshow(f"{name}", cv.vconcat([cv.hconcat(bigs), cv.hconcat(row2)]))

    return thr[ind]

def deluge_qr(imgfilename, method=BOTH, dbg=True):
    """
    Reads deluge crash handler patterns from image
    files.
    """
    return deluge_qr_img(cv.imread(imgfilename), method=method, dbg=dbg)

def deluge_qr_url(url, method=BOTH, dbg=False):
    """
    Grab an image from the given URL and attempt to read a deluge
    crash handler pattern.
    """
    response = requests.get(url)
    image_bytes = response.content
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    code, img = deluge_qr_img(image, method=method, is_bot=True)
    return code, img


def deluge_qr_img(img, method=BOTH, dbg=False, is_bot=False):
    """
    Try both detection methods in turn to detect a deluge QR crash pattern
    """
    if method == BOTH:
        methods = [GRID, HOUGH]
    else:
        methods = [method]

    for method in methods:
        try:
            code, comp = deluge_qr_img_method(img, method=method, dbg=dbg)
            if len(code) == 5:
                if all([addr == 0 or 
                    (addr > 0x20000000 and
                     addr < 0x30000000) for addr in code[:4]] ):
                    return code, comp
                if not is_bot:
                    # If we're running as the discord bot, only return
                    # codes that look like deluge decodes, starting 0x20
                    # Otherwise return whatever code we read
                    return code, comp
        except Exception as e:
            if not is_bot:
                raise e
            else:
                pass

    raise Exception("Failed to read a Deluge Crash pattern from the image")

def deluge_qr_img_method(img, method=GRID, dbg=False):
    """
    Identifies the deluge button grid in a photo and reads
    off the crash handler pointers and github commit hash.
    """
    imc = img

    im = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    im = (255-im)

    # Scale down to approx 800 px image width/height
    h, w = im.shape[:2]
    scale = max(1.0, h / 800.0, w / 800.0)

    h = h // int(scale)
    w = w // int(scale)
    im = cv.resize(im, (w, h), interpolation= cv.INTER_LINEAR)
    #im = cv.bilateralFilter(im, 7, 21, 7)
    #im = cv.medianBlur(im, 5)

    imc = cv.resize(imc, (w, h), interpolation= cv.INTER_LINEAR)
    imdbg = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    imc_clean = imc.copy()
    
    qrdebug.img_clean = imc.copy() 
    # Lower and upper bound on deluge button sizes, assuming the deluge takes up most of the frame.
    bwmax = max(w,h) // 20
    bwmin = bwmax // 3.5

    skp, pads, gridpads, dots, imc = locate_grid(im, imc, bwmin, bwmax)

    d2 = dots.copy()
    d2 = cv.drawKeypoints(d2, skp, 0, (0, 0, 255), 
                                 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    n = len(skp)
    maxd = bwmax * 1.2
    median_size = [kp.size for kp in skp][n//2]
    isz = int(median_size)//3
    
    grida = np.array([kp.kp.pt for kp in gridpads])
    gridca = np.cov(grida, y=None, rowvar = 0, bias= 1)

    v, vect = np.linalg.eig(gridca)
    tvect = np.transpose(vect)

    qrdebug.draw_cross(imc, w//2, h//2, tvect)

    if dbg:
        cv.imshow("aft DMF", imc)
        cv.imshow("Thresholded Pads", dots)
        cv.imshow("TDD", d2)

    if method == HOUGH:
        # Try finding the grid by looking for linear rows of possible pads
        # This works well if there are many gaps to jump over that would
        # disrupt the grid method, but is more susceptible to being thrown
        # off by spurious keypoints that are off the grid.
        hough.hough_grid( w,h, pads)
        grpmax = 1
    else:
        # Try finding the grid by looking for keypoints that form gridlike
        # patterns. This relies on being able to distinguish both the lit 
        # and unlit pads on the deluge. In dark images, or those using a
        # faceplate overlay that isn't black, the hough method above may
        # better.
        for kp in gridpads:
            orient_to_grid(pads, kp, vect)

        grpmax = flood_fill_uv_grid(pads, gridpads)

    grppnts = [gp for gp in gridpads if gp.group == grpmax]
    minu, minv, maxu, maxv = extents(grppnts)

    if method == HOUGH and maxu-minu > 12:
        # Trim both edges off the potential grid, to avoid the
        # status / audition pads, regardless of orientation.
        grppnts = [gp for gp in gridpads if gp.group == grpmax and gp.u>minu+3 and gp.u < maxu-3] 
        minu, minv, maxu, maxv = extents(grppnts)
   
    if dbg:
        qrdebug.draw("I", imdbg, pads, lambda k:str(k.i))
        qrdebug.draw("U", imdbg, pads, lambda k:str(k.u if k.u is not None else " "))
        qrdebug.draw("V", imdbg, pads, lambda k:str(k.v if k.u is not None else " "))
    
    # Find the distortion between this (possibly partial) grid and the image
    hmg = homography_from_partial_grid(grppnts)

    invh = np.linalg.pinv(hmg)

    missing_u = 15-(maxu-minu)
    missing_v = 15-(maxv-minv)

    # Cast a wider grid of u/v (pad grid) and x/y (pixel) coordinates, making
    # it wide enough to hold 18*6 pads, regardless of which direction they lie in
    # and which parts of the grid we have already found.
    wide_uvgrid = build_grid(minu-missing_u,maxu+missing_u,minv-missing_v,maxv+missing_v) 
    grid = cv.perspectiveTransform(np.array([wide_uvgrid.reshape((-1,2))]), invh).reshape(wide_uvgrid.shape)
    if dbg:
        imcgrid = imc.copy()
        qrdebug.draw_uvgrid(imcgrid, grid, wide_uvgrid, coords=False)
        cv.imshow("sdfkjl", imcgrid)

    maxu -= int(wide_uvgrid[0][0][0])
    maxv -= int(wide_uvgrid[0][0][1])
    minu -= int(wide_uvgrid[0][0][0])
    minv -= int(wide_uvgrid[0][0][1])

    if dbg:
        qrdebug.draw("U", imdbg, pads, lambda k:str(k.u if k.u is not None else " "))
        qrdebug.draw("V", imdbg, pads, lambda k:str(k.v if k.u is not None else " "))
        qrdebug.draw("I", imdbg, pads, lambda k:str(k.i))
    
    # Push the grid out into other parts of the image and snap it onto
    # the keypoints found there
    limits = [maxu, maxv, minu, minv]  
    limits, grid, hmg = expand_grid(pads, wide_uvgrid, grppnts, limits, grid, skp, maxd)
    maxu, maxv, minu, minv = tuple(limits)

    uvgrid = wide_uvgrid[minu:maxu+1, minv:maxv+1,:] 
    xygrid = grid[minu:maxu+1, minv:maxv+1,:]

    if dbg:
        imblk2 = np.zeros(imc.shape, dtype=np.uint8)
        qrdebug.draw_uvgrid(imblk2, xygrid, uvgrid, coords=True)
        
        qrdebug.draw("G2", imdbg, grppnts, lambda k:str(k.i))
        cv.imshow("G2", imdbg)
    
    if maxu-minu < maxv-minv:
        uvgrid = np.flip(np.transpose(uvgrid, axes=(1,0,2)), axis=2)
        xygrid = np.transpose(xygrid, axes=(1,0,2))

    uvgrid[:,:,0] -= uvgrid[0,0,0]
    uvgrid[:,:,1] -= uvgrid[0,0,1]

    corner_uvs = [(0,0), (15,0), (15,7), (0,7)]
    corner_colours = [(0,92,255),(0,255,192), (192, 200, 0), (255, 92, 0)]
    
    for i in range(4):
        u,v = corner_uvs[i]
        circle(imc, xygrid[u][v], 15, corner_colours[i], thickness=-1)
    
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

    if nleft > nright:
        xygrid = np.flip(xygrid, axis=(0,1))
        sbgrid = np.flip(sbgrid, axis=(0,1))

    if method == GRID:
        if maxu-minu < maxv-minv:
            xygrid = np.flip(xygrid, axis=1)
            sbgrid = np.flip(sbgrid, axis=1)
    else:    
        vec = xygrid[1][0] - xygrid[0][0] 
        vecr = [vec[1], -vec[0]]
        vecv = xygrid[0][1] - xygrid[0][0] 
        if (np.dot(vecr, vecv) > 1.0):
            xygrid = np.flip(xygrid, axis=1)
            sbgrid = np.flip(sbgrid, axis=1)

    imblk = np.zeros(imc.shape, dtype=np.uint8)
    qrdebug.draw_uvgrid(imblk, xygrid, uvgrid, coords=True)
    qrdebug.draw_uvgrid(imblk, sbgrid, uvgrid, coords=True)
  
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

    isz = isz //2
    for i in range(16):
        for j in range(8):
            x = int(xygrid[i][j][0])
            y = int(xygrid[i][j][1])
            im4 = im[y-isz:y+isz, x-isz:x+isz]
            pads_gray[j][i] = int(cv.mean(im4)[0])
            im4 = imc_clean[y-isz:y+isz, x-isz:x+isz]
            col = [int(ii) for ii in cv.mean(im4)[:3]]
            pads_bgr[j,i,:] = col
            cv.rectangle(imc, (x-isz, y-isz), (x+isz, y+isz), (0,0,255), 1)
            
    for i in range(2):
        for j in range(8):
            x = int(sbgrid[i][j][0])
            y = int(sbgrid[i][j][1])
            im4 = im[y-isz:y+isz, x-isz:x+isz]
            pads_gray[j][i+16] = int(cv.mean(im4)[0])
            im4 = imc_clean[y-isz:y+isz, x-isz:x+isz]
            col = [int(ii) for ii in cv.mean(im4)[:3]]
            pads_bgr[j,i+16,:] = col
            cv.rectangle(imc, (x-isz, y-isz), (x+isz, y+isz), (0,0,255), 1)

    out = [0]*18

    quarters = [pads_bgr[:,4*x:4*x+4] for x in range(4)]
    sidebar = pads_bgr[:,-2:]

    pieces = []
    for i, q in enumerate(quarters):
        pieces.append(threshold_grid_section(f"PADS{i}", q, dbg))

    pieces.append(threshold_grid_section(f"SIDE0", sidebar, dbg))

    pads_v_threshold = cv.hconcat(pieces)
        
    for ix in range(18):
        for iy in range(8):
            if pads_v_threshold[iy][ix] > 128:
                out[ix] += (0x1 << iy)
    print("")
    for iy in range(8):
        for ix in range(18):
            if ix == 16:
                print(" | ", end="")
            if pads_v_threshold[iy][ix] > 128:
                print("[]",end="")
            else:
                print("  ",end="")
        print("")

       
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

    #pads_gray_big = cv.resize(pads_gray, (18*32, 8*32), interpolation= cv.INTER_NEAREST)
    pads_bgr_big = cv.resize(pads_bgr, (18*32, 8*32), interpolation= cv.INTER_NEAREST)
    #pads_v_big = cv.resize(pads_v, (18*32, 8*32), interpolation= cv.INTER_NEAREST)
    #pads_v_thr_big = cv.resize(pads_v_threshold, (18*32, 8*32), interpolation= cv.INTER_NEAREST)

    pads_shifted = np.zeros((8, 19), dtype=np.uint8)
    pads_shifted[0:8,0:16] = pads_v_threshold[0:8,0:16]
    pads_shifted[0:8,17:19] = pads_v_threshold[0:8,16:18]
    
    overlay = cv.warpPerspective(pads_shifted, invh, (w, h), flags= cv.INTER_NEAREST)
    overlay = cv.cvtColor(overlay, cv.COLOR_GRAY2BGR)
    comp = cv.bitwise_or(imc, overlay)
    #pads_v_big = cv.resize(pads_v, (18*32, 8*32), interpolation= cv.INTER_NEAREST)
    
    if dbg:
        cv.imshow("Keypoints", imc)    
        cv.imshow("RGB", pads_bgr)
        cv.imshow("Pads bgr big", pads_bgr_big)
        
        cv.imshow("Grey", imdbg)

        cv.imshow("m", comp)
        
        cv.imshow("Clean", imc_clean)
        cv.waitKey(0)

    return f, comp

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Command line argument parser example')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable verbose debug output')
    parser.add_argument('-m', '--method', choices=['grid', 'hough', 'both'],
                         default='both', help='Choose grid detection method')
    parser.add_argument('filename', type=str, help='Image to process')
    args = parser.parse_args()

    methods = {'grid':GRID, 'hough':HOUGH, 'both':BOTH}

    try:
        code, comp = deluge_qr(args.filename, method=methods[args.method], dbg=args.debug)
        commit_fragment = f"{code[4]:04x}"
        st = f"0x{commit_fragment}"
        print(st)
 
    except Exception as e:
        # Pause so we can inspect the debug images.
        cv.waitKey(0)
        raise e
