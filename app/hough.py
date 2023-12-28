import cv2 as cv
import numpy as np
import math
import qrdebug
from qrdebug import circle
from dataclasses import dataclass, astuple

@dataclass
class Line:
    """
    A line of keypoints found in the image
    """
    r: float                  # Radius (distance from origin)
    theta: float              # Angle in radians from X-axis
    ids: list[int]            # Indices of keypoints on this line
    ts: list[float]           # T-values along this line
    us: list[float] = None    # Rounded U values along this line
    ud: list[float] = None    # Discrepancies from integer fitting
    uf: float = None          # Scale factor from u space to t space
    ufo: float = None         # Offset in t.


def hough_grid(w, h, pads):
    """
    Alternative grid detection method.
    Performs a hough transform on the list of potential pads.
    This is effectively a brute force search for all groups of points
    that form a line. We look for up to 8 lines with the most points, hoping
    that the 16 grid pads + 2 audition pads will be the dominant linear features.
    """
    res = min([min(p.dists) for p in pads if p.dists!=[]])

    xs = np.array([kp.kp.pt[0] for kp in pads])
    ys = np.array([kp.kp.pt[1] for kp in pads])
    
    nr = 500
    maxr = np.sqrt(w**2 + h**2)

    dr = (2*maxr) / nr

    ntheta = 300

    hough = np.zeros((nr, ntheta))

    for itheta in range(ntheta):
        theta = itheta * math.pi / ntheta
        pads_dist = - xs * np.cos(theta) - ys * np.sin(theta)
        minp = np.min(pads_dist)
        maxp = np.max(pads_dist)
        for gp in pads:
            r = pads_dist[gp.i]
            ir = int((maxr+r) / dr)
            c = np.sum(np.abs(pads_dist - r) < res/10)

            if c > 3:
                hough[ir][itheta] += float(c)

    hough /= np.max(hough)
    #cv.imshow("HHH", enlarge(hough,3))

    image = qrdebug.img_clean.copy()

    hc = hough.copy()
    hc2 = hough.copy()  
    ang = None
    lines = []
    for i in range(8):
        mx = np.unravel_index(np.argmax(hc), hc.shape)

        mxr = (mx[0]*dr) - maxr
        mxtheta = (mx[1]*math.pi/ntheta)

        r = mxr
        theta = mxtheta# + math.pi

        setrect(hc,mx[1],mx[0],4,2,0.0)
        setrect(hc2,mx[1],mx[0],4,2,1.0)
        
        if ang is None:
            ang = theta
        else:
            dang = abs(ang-theta)
            if not (abs(dang) < 0.2 or abs(dang-math.pi) < 0.2):
                continue

        if r > 0.0:
            r = -r
            theta -= math.pi

        pads_dist = - xs * np.cos(theta) - ys * np.sin(theta)
        
        ids = [i for i in range(len(pads)) if (abs(pads_dist[i]-r)< res/2)]

        dup = False
        for line in lines:
            for id in line.ids:
                if id in ids:
                    dup = True
        if dup:
            continue

        line = Line(r, theta, [], [])
        ts = [line_project(line, xs[i], ys[i]) for i in ids]
        
        pts = list(zip(ids, ts))
        pts.sort(key=lambda x:x[1])
        
        ids, ts = zip(*pts)
        
        line.ids = ids
        line.ts = ts       
        lines.append(line)
        draw_line(image, line)
        

    lines.sort(key=lambda x:x.r)
    
    match_lines(lines,pads)
    
    #cv.imshow("HHH2", enlarge(hc,1))
    #cv.imshow("HHH3", enlarge(hc2,1))
    #cv.imshow("line", image)

    #cv.waitKey(0)
    
    return lines

def setrect(image, x, y, xb, yb, val):
    """
    Paints over a subrectangle of the hough transform to remove
    a local maximum and surrounding area.
    """
    height, width = image.shape
    c = [x,y]
    b = [xb,yb]
    rng = [None,None]
    # Calculate the coordinates of the top-left and bottom-right corners of the rectangle
    for crd in [0,1]:
        lw = int(c[crd] - b[crd])
        hi = int(c[crd] + b[crd])
        mx = image.shape[1-crd]
        if lw < 0:
            rng[crd] = [[0,hi],[mx+lw,mx]]
        elif hi > mx:
            rng[crd] = [[lw,mx],[0,hi-mx]]
        else:
            rng[crd] = [[lw,hi]]
    for xr in rng[0]:
        for yr in rng[1]:
            image[yr[0]:yr[1], xr[0]:xr[1]] = val

def line_eval(line, t):
    """
    Maps a t-value along a given line to x/y pixel coordinates
    """
    r = line.r
    theta = line.theta
    v = - t *np.array([-np.sin(theta), np.cos(theta)]) \
      - r * np.array([np.cos(theta), np.sin(theta)])
    return v[0], v[1]

def line_project(line, x, y):
    """
    Finds the T-value on a line for the closest approach from the given x/y
    """
    theta = line.theta
    return x *np.sin(theta) - y*np.cos(theta)


def match_lines(lines, pads):
    """
    Given a list of 'lines', that we hope represent the rows
    of the deluge pad grid, parameterise the pads in U and V
    as best we can.
    """
    imgm = qrdebug.img_clean.copy()

    # Assign v-coordinates based on line spacing.
    rs = [l.r for l in lines]
    vs, _, _, _ = fit_integer(rs,8)

    toff = 0
    for i, line in enumerate(lines):
        ts = line.ts
        line.us, line.ud, line.uf, line.ufo = fit_integer(ts,20)

    med_factor = [line.uf for line in lines][len(lines)//2]
    lines = [line for line in lines if abs(line.uf-med_factor)/med_factor < 0.05]

    rs = [l.r for l in lines]
    vs, _, _, _ = fit_integer(rs,8)

    for i, line in enumerate(lines):
        #draw_annotated_line(imgm,line)
        if i > 0:
            udm1 = lines[i-1].ud
            for j in range(len(udm1)):
                if udm1[j] < 0.15:
                    tcmp = lines[i-1].ts[j]
                    joff = lines[i-1].us[j]
                    break
                  
            x,y = line_eval(lines[i-1], tcmp)
            circle(imgm, (x,y), 6, (0,0,255))
            tn = line_project(line, x, y)
            x,y = line_eval(lines[i], tn)
            circle(imgm, (x,y), 6, (0,255,0))
            toff += (int(np.round((tn - line.ufo)/line.uf))  - joff)


        for j,id in enumerate(line.ids):
            if line.ud[j] < 0.15:
                pads[id].u = int(line.us[j]-toff)
                pads[id].v = int(7-vs[i])
                pads[id].group = 1
    #cv.imshow("ANNOT", imgm)
    #cv.waitKey(0)

def fit_integer(flts, mx):
    match = np.array(flts)
    sbest = 1000.0
    rbest = None
    dbest = None
    fbest = None
    obest = None
    for ileft in range(0,len(flts)):
        for iright in range(len(flts)-1, ileft+1, -1):
            imin = max(1,iright-ileft - 2)
            imax = mx
            #print(imin, imax)
            for factor in range(imin,imax+1):
                m2 = (match-match[ileft])
                div = (m2[iright] / factor)
                m2 /= div
                rr = np.round(m2)  
                if rr[-1]-rr[0] > mx:
                    continue
                c = np.sum(np.abs(rr - m2) < 0.1)
                sqdr = np.sum((rr-m2)**2.0)
                if sqdr < sbest:
                    sbest = sqdr
                    fbest = div
                    obest = match[ileft] + rr[0]*fbest
                    rbest = [r - rr[0] for r in rr]
                    dbest = np.abs(rr - m2)
                #print("    B:", ileft, iright, c, factor, sqdr)
                #print("    ", [f"{v:.2f}" for v in m2])
                #print("    ",[f"{v:.2f}" for v in np.abs(rr - m2)])
    return rbest, dbest, fbest, obest

def draw_annotated_line(img, line):
    draw_line(img,line)
    for i in range(18):
        t = (i-0.5)*line.uf + line.ufo
        circle(img, line_eval(line,t), 2, (128,0,255))

def draw_line(img, line):
    r, theta, ids, ts, _, _, _, _ = astuple(line)
    x1,y1 = [int(i) for i in line_eval(line, ts[0])]
    x2,y2 = [int(i) for i in line_eval(line, ts[-1])]
    
    cv.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
    
    for t in ts:
        circle(img, line_eval(line, t), 6, (255,255,0))
