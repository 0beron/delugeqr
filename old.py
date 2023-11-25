   kpa2 = []
    maxd = median_dist*1.6
    low = 0
    hi = 0

    #print(median_dist)
    cv.circle(imdbg, (int(skp[0].pt[0]), int(skp[0].pt[1])), int(maxd), (0, 0, 255), thickness=2, lineType=cv.LINE_AA)

    if dbg == 1:
        cv.circle(imdbg, (w//2,h//2), int(bwmin), (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
        cv.circle(imdbg, (w//2,h//2), int(bwmax), (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
 
    filtered_keypoints = [nk for nk in pads if len(nk.neighbours) >= 7 and nk.valid ]

    imc = cv.drawKeypoints(imc, skp, np.array([]), (255,255,0), 0)
    for i in range(n):
        cv.circle(imc, (int(skp[i].pt[0]), int(skp[i].pt[1])), isz, (bris[i], bris[i], bris[i]), thickness=2, lineType=cv.LINE_AA)

    draw("nngh", imdbg, filtered_keypoints, lambda k:str(len(k.neighbours)))
    draw("index", imdbg, pads, lambda k:str(k.i))
    
    old_filtered_keypoints = [nk.kp for nk in pads if len(nk.neighbours) >= 7 ]
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




  #  ------------------------------------------------------------------------
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

