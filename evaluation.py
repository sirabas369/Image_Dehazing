import cv2
import skimage
import numpy as np
import math
from scipy.interpolate import UnivariateSpline

def LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

data = np.zeros((16,3))

for idx in range(1,17):
    if (idx<10 and idx != 3):
        s1 = '/home/sirabas/Documents/'+'0'+str(idx)+'_outdoor_pdh.png'
        s2 = "/home/sirabas/Documents/AGV/dehazing/Ground Truth/"+'0'+str(idx)+"_outdoor_GT.jpg"
    elif (idx == 3):
        s1 = '/home/sirabas/Documents/'+'0'+str(idx)+'_outdoor_pdh.png'
        s2 = "/home/sirabas/Documents/AGV/dehazing/Ground Truth/"+'0'+str(idx)+"_outdoor_GT.JPG" 
    else:
        s1 = '/home/sirabas/Documents/' + str(idx) + '_outdoor_pdh.png'
        s2 = "/home/sirabas/Documents/AGV/dehazing/Ground Truth/" + str(idx) + "_outdoor_GT.jpg"


    dehazed = cv2.imread(s1,1)
    truth = cv2.imread(s2,1)

    h = truth.shape[0] // 3
    w = truth.shape[1] // 3
    dim = (w, h)
    truth = cv2.resize(truth, dim, interpolation=cv2.INTER_AREA)
    plain_dehazed = dehazed.astype(np.uint8)

    #clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(15,15))

    hsv = cv2.cvtColor(plain_dehazed, cv2.COLOR_BGR2HSV)

    #hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 90, 160, 256])
    hue, saturation, value  = cv2.split(hsv)
    saturation = cv2.LUT(saturation, increaseLookupTable).astype(np.uint8)
    contrast = cv2.merge((hue, saturation, value ))

    #for i in range(hsv.shape[0]):
    #    for j in range(hsv.shape[1]):
    #        if(hsv[i, j, 1]*1.5 <= 255):
    #            hsv[i, j, 1] = (hsv[i, j, 1])*1.5

    res = cv2.cvtColor(contrast, cv2.COLOR_HSV2BGR)

    gam_cor = adjust_gamma(res, 1.15)

    res1 = cv2.detailEnhance(gam_cor, sigma_s=3, sigma_r=0.05)

    psnr1 = psnr(truth.astype(np.uint8), res1)
    ssim = skimage.metrics.structural_similarity(truth.astype(np.uint8), res1,multichannel=True)
    print(idx,' psnr , ssim = ', psnr1 , ssim)

    cv2.imshow('truth', truth)
    cv2.imshow('improved', res1)

    data[idx-1] = idx, psnr1, ssim
    #cv2.imwrite(f'/home/sirabas/Documents/AGV/dehazing/outputimages2/{idx : 02}_outdoor_dh.png', res1)

    cv2.waitKey(0)

#np.savetxt("psnr_ssim_values2.csv", data, delimiter=',')