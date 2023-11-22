import cv2 as cv
import numpy as np
import math
import sys

from dataclasses import dataclass


def kpdst(kp1, kp2):
    dx = kp1.pt[0]-kp2.pt[0]
    dy = kp1.pt[1]-kp2.pt[1]
    return round(math.sqrt(dx*dx + dy*dy), 2)

def update_xrange(low, hi, i, kp, dstmax):
    while(kp[i].pt[0] - kp[low].pt[0] > dstmax):
        low += 1
    while(hi < len(kp) and kp[hi].pt[0] - kp[i].pt[0] < dstmax):
        hi += 1
    return low, hi

@dataclass
class Keypoint:

    kp: object
    neighbours: list[int]
    dists: list[float]
    

def threshold_erode(im, thr):
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

def reject_silk_screen(im, bwmin, bwmax):
    mblobs = 0
    for t in range(10, 255, 10):
        dots = threshold_erode(im, t)
        blobs = find_blobs(dots, bwmin, bwmax, binary=True)
        if len(blobs) > mblobs:
            mblobs = len(blobs)
            print("mmmm ", t)
            imblobs = dots
            bmax = blobs
        print(t, len(blobs))
        break
    return bmax

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
        nkp.append(Keypoint(skp[i], js,ds))
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
        im4 = imc[y-isz:y+isz, x-isz:x+isz]
        col = [int(ii) for ii in cv.mean(im4)[:3]]
        cols.append(col)

    kpa2 = []
    maxd = median_dist*1.6
    low = 0
    hi = 0

    print(median_dist)
    cv.circle(imdbg, (int(skp[0].pt[0]), int(skp[0].pt[1])), int(maxd), (0, 0, 255), thickness=2, lineType=cv.LINE_AA)

#    for i in range(n):
#        low, hi = update_xrange(low, hi, i, skp, maxd)
#        d = w*2
#        nngh = 0
#        minpt = skp[i].pt
#        maxpt = skp[i].pt
#        bx = None
#        for j in range(low, min(hi, n)):
#            if i==j:
#                continue
#            if abs(skp[i].pt[1] - skp[j].pt[1]) < maxd:
#                dst = kpdst(skp[i], skp[j])
#                if (dst < maxd):
#                    nngh += 1
#                    minpt = (min(skp[j].pt[0], minpt[0]), min(skp[j].pt[1], minpt[1]))
#                    maxpt = (max(skp[j].pt[0], maxpt[0]), max(skp[j].pt[1], maxpt[1]))
#                d = min(d, dst)
#                bx = maxpt[0]-minpt[0]
#                by = maxpt[1]-minpt[1]###

#        if bx is not None:
#            kpa2.append((skp[i], d, nngh, bx, by))
#            if nngh > 6:
#                c = (0,0,255)
#            else:
#                c = (192,192,192)

    if dbg == 1:
        cv.circle(imdbg, (w//2,h//2), int(bwmin), (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
        cv.circle(imdbg, (w//2,h//2), int(bwmax), (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
            
#    if len(kpa2) == 0:
#        cv.imshow("Nothing", imdbg)
#        cv.waitKey(0)
#        sys.exit(1)
            
        
#    kp, ds, ng, bxs, bys = zip(*kpa2)

#    sz = [k.size for k in nkp]
#    med = sz[len(sz)//2]

    #for i in range(len(kpa2)):
    #    kp = kpa2[i]
    #    print(i, kp)

#    print(med*1.2)

    filtered_keypoints = [nk for nk in nkp if len(nk.neighbours) >= 7 ]# and
#           kp[3] > med*1.2 and
#           kp[4] > med*1.2]

    #imc = cv.drawKeypoints(imc, skp, np.array([]), (255,255,0), 0)
    #for i in range(n):
    #    cv.circle(imc, (int(skp[i].pt[0]), int(skp[i].pt[1])), isz, (bris[i], bris[i], bris[i]), thickness=2, lineType=cv.LINE_AA)

    for kp in filtered_keypoints:
        x = int(kp.kp.pt[0])
        y = int(kp.kp.pt[1])
        nngh = len(kp.neighbours)
        if nngh > 6:
            c = (0,0,255)
        else:
            c = (192,192,192)
        cv.putText(imdbg, str(nngh), (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, c)

    filtered_keypoints = [nk.kp for nk in nkp if len(nk.neighbours) >= 7 ]
    imc = cv.drawKeypoints(imc, filtered_keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    if len(filtered_keypoints) < 4:
        print("Too few blobs remained after filtering")
        # Show keypoints
        if dbg == 1:
            cv.imshow("Grey", imdbg)
            cv.imshow("Keypoints", imc)
            cv.waitKey(0)
        sys.exit(1)

    a_all = np.array([kp.pt for kp in skp])
    a = np.array([kp.pt for kp in filtered_keypoints])
    
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
