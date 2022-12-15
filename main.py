import numpy as np
import cv2

def dcp(img, rsize, dimension):
    global h, w
    dcp = np.zeros((h, w), np.uint8)
    if dimension == 2:
        dcp = np.empty((h, w), dtype=float)
    
    psize = (rsize-1)//2
    for i in range(0, h):
        for j in range(0, w):
            if (dimension == 3):
                ker = np.full((rsize, rsize, 3), 255)
            if dimension == 2:
                ker = np.full((rsize, rsize, 3), 255.0)

            for k in range(-psize, psize+1):
                for l in range(-psize, psize+1):
                    if ((i+k >= 0) and (j+l >= 0)) and ((j+l < w) and (i+k < h)):
                        ker[psize+k, psize+l] = (img[i+k, j+l])

            minval = np.min(ker)

            dcp[i, j] = minval
    return dcp


def findA(img, dcp):
    global h, w
    imggrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    linlisdcp = []
    linlisimg = []
    linlisimggrey = []
    for i in range(h):
        linlisdcp.extend(dcp[i])
        linlisimg.extend(img[i])
        linlisimggrey.extend(imggrey[i])
    l = np.argsort(linlisdcp)
    l = l[(len(l)*999)//1000 : ]
    maxpos = []
    maxint = 0
    for i in l:

        if ((maxint < linlisimggrey[i])):
            maxpos = i
            maxint = linlisimggrey[i]

    Abgr = linlisimg[maxpos]
    return Abgr


def findt(img, a):
    global h,w
    imgnormed = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            for k in range(3):
                imgnormed[i, j, k] = img[i, j, k]/a[k]
    normeddcp = dcp(imgnormed, 15, 2)
    t = 1- 0.85*normeddcp
    return t


def guidedfilter(img, p):
    r = 20
    e = 1e-06
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imggrey = np.float64(image)/255

    meani = cv2.boxFilter(imggrey, cv2.CV_64F, (r,r) )
    meanp = cv2.boxFilter(p, cv2.CV_64F, (r,r) )
    meanip = cv2.boxFilter(imggrey*p, cv2.CV_64F, (r,r) )
    covip = meanip - meani*meanp
    meanii = cv2.boxFilter(imggrey*imggrey, cv2.CV_64F, (r,r) )
    vari = meanii - meani*meani
    print(vari)
    a = covip / (vari + e)
    b = meanp - a*meani

    meana = cv2.boxFilter(a, cv2.CV_64F, (r,r) )
    meanb = cv2.boxFilter(b, cv2.CV_64F, (r,r) )

    q = meana*imggrey + meanb
    return q

haze=cv2.imread("/home/sirabas/Documents/AGV/dehazing/Hazy Images/16_outdoor_hazy.jpg",1)
truth=cv2.imread("/home/sirabas/Documents/AGV/dehazing/Ground Truth/16_outdoor_GT.jpg",1)
h = haze.shape[0] // 3
w = haze.shape[1] // 3

dim = (w, h)
haze = cv2.resize(haze, dim, interpolation=cv2.INTER_AREA)
truth = cv2.resize(truth, dim, interpolation=cv2.INTER_AREA)

print("Finding DCP")
dcp1 = dcp(haze, 15, 3)
print("Dcp done")
print("Finding A")
a = findA(haze, dcp1)
print("Afound")
print("Finding T")
t = findt(haze, a)
print("T found")
print("Applying Guided Filter on T")
tfinal=guidedfilter(haze, t)

for i in range(h):
        for j in range(w):
            if (tfinal[i, j] < 0.1):
                tfinal[i, j] = 0.1


print("Making final image")
cv2.imshow("t", (t*255).astype(np.uint8))
cv2.imshow("tfinal", (tfinal*255).astype(np.uint8))
dehazed = np.zeros((h, w, 3))
for i in range(h):
    for j in range(w):
        for k in range(3):
            dehazed[i, j, k] = (haze[i, j, k]-a[k]*(1-tfinal[i, j]))/tfinal[i, j]


plain_dehazed = dehazed.astype(np.uint8)

cv2.imshow("haze", haze.astype(np.uint8))
cv2.imshow("dehazed image", plain_dehazed)
cv2.imshow("org", truth.astype(np.uint8))
#cv2.imwrite("16_outdoor_pdh.png", plain_dehazed)
#cv2.imwrite("16_outdoor_t.png", (t*255).astype(np.uint8))
#cv2.imwrite("16_outdoor_tfinal.png", (tfinal*255).astype(np.uint8))
cv2.waitKey(0)