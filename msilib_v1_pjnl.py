#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The msilib_v1_pjnl.py contains two funtions to automatic segmentation of the area that containt the grapes in
multispectral images

1. The function_ extract_pattern(folder, namePattern) allows to generate binary patterns from multispectral grapes images. A binary patters is composed by the binary image with the edges of the grape.
Call Example:

    msi_folder = '../dataset/Noviembre2020_Negras/Uva_Negra1_25.11.20/IR/UVA1/autum_royal_32a.png'
    namePattern = 'binpattern1.png'
    extract_pattern(folder, namePattern):

2. The function segmentation(msi_folder,patternname,visible=False) carries out the automatic cropping of area of the msi grapes (located in /msi_folder) based on the binary patters (located in 'pathpattern1')

Call Example:
    pathpattern1='../patternbins_roja_ir_16dic2020/'
    patternbinfile=os.path.join(pathpattern1,'binpattern1'+'.png')
    msi = msilib_pjnl.segmentation(path, patternbinfile)

@author: pedrojaviernavarrolorente
"""
import matplotlib.pyplot as plt
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
import cv2
import os
import numpy as np
import skimage.measure  
import math
import imutils


def extract_pattern(folder, namePattern):
    images=[]
    entropy = []

    filesnames = os.listdir(folder)
    filesnames.sort()
    N, = np.array(filesnames).shape
    for i in range(N):
        img = cv2.imread(os.path.join(folder,filesnames[i]),0)#, dtype = 'float')
    
        if img is not None:
            # Append all the images
            images.append(img)
            entropy.append(skimage.measure.shannon_entropy(img))

    
    umbral_entropy= np.array(entropy).mean()
    N, m ,n = np.array(images).shape 
    img_bw=[]
    # print(' Bands number %i'%(N))
    for i in range(N):
        if entropy[i] > umbral_entropy:
            # image noisy -> Ad-hoc threshold
            ret,th1 = cv2.threshold(images[i],10,255,cv2.THRESH_BINARY) 

        else:  
            # image non noisy  -> Otsu threshold
            ret,th1 = cv2.threshold(images[i],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        img_bw.append(th1) # append all threshold images 

    N, m ,n = np.array(images).shape 
    
    # Segmentation process
    
    areas_max_contorno=[]
    ind_contorno_areamax = []
    umbralarea_min = 9000 # minimum countours area
    umbralarea_max = 45000 # maximum countours area
    for nbanda in range(N):
        i=0
        areas = []
        ncontorno = []

        contours, hierarchy = cv2.findContours(img_bw[nbanda], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            # Area filters
            if umbralarea_min <area < umbralarea_max:
                circularity = 4*math.pi*(area/(perimeter*perimeter))
                # circularity filters
                if 0.4 < circularity < 1.2:
                    areas.append(area)
                    ncontorno.append(i)

                else:
                    areas.append(0)
                    ncontorno.append(i)

            else:
                areas.append(0)  
                ncontorno.append(i)
                
            i=i+1
        # take the maximum area contour
        areas_max_contorno.append(max(areas)) 
        ind_contorno_areamax.append(ncontorno[np.argmax(areas)])

        
  
    mejor_banda = np.argmax(areas_max_contorno)
    mejor_contorno = ind_contorno_areamax[mejor_banda]
    print('Best segmentation on band %i contour %i'%(mejor_banda, mejor_contorno))
    
    contours, hierarchy = cv2.findContours(img_bw[mejor_banda], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # ----------------------------------------------------------------     
    # Show the image with the drawn contours
    # ----------------------------------------------------------------     
    k = 2
    x,y,w,h = cv2.boundingRect(contours[mejor_contorno])
    if (y-k < 0) or (x-k < 0):
        k=0
        print('CORRECT ROI: %i:%i,%i:%i ',(y-k,y+h+k, x-k,x+w+k))
        
    Ibin = img_bw[mejor_banda]
    img = images[mejor_banda]
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    
    imgplot = plt.imshow(img,'gray')
    ax.set_title('BANDA '+str(mejor_banda))
    ROI = img[y-k:y+h+k, x-k:x+w+k]

    
    ax = fig.add_subplot(1, 3, 2)
    imgplot = plt.imshow(ROI,'gray')
    ax.set_title('segmentation')
    
    # Filter using contour area and remove small noise
    
    cnts = cv2.findContours(Ibin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    thresh1 = np.zeros(img.shape, np.uint8)
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 2000:
            cv2.drawContours(thresh1, [c], -1, (255,0,0), -1)
            # print(area)
    
            
    kernel = np.ones((5, 5), np.uint8)        
    thresh = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=1) 
    
    # Smooth edges 
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    Ibin = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    
    
    binpattern = Ibin[y-k:y+h+k, x-k:x+w+k]
    ax = fig.add_subplot(1, 3, 3)
    imgplot = plt.imshow(binpattern,'gray')
    ax.set_title('bin_pattern')
    
    # Save pattern 
    cv2.imwrite(namePattern +'.png', binpattern)
    
    imgplot = plt.imshow(binpattern,'gray')
    
    plt.show()
    plt.close()


def segmentation(folder,patternname,visible=False):
    # ----------------------------------------------------------------------------
    # Visualization parameters
    # ----------------------------------------------------------------------------
    nf=3
    nc=4
    Titles = [488.38, 488.58, 503.59, 516.60, 530.62, 542.17, 567.96, 579.29, 592.89, 602.88, 616.59, 625.71]
    images = []
    filesnames = os.listdir(folder)
    filesnames.sort()
    N, = np.array(filesnames).shape
    for i in range(N):
        img = cv2.imread(os.path.join(folder,filesnames[i]),0)#, dtype = 'float')
        if img is not None:
            images.append(img)
            
    founds = []
    for band in range(0,N):
        img = images[band]
        # print('Imagen %i-->banda: %i'%(Nimage,band))
        def adjust_gamma(image, gamma=1.0):
        	# build a lookup table mapping the pixel values [0, 255] to
        	# their adjusted gamma values
        	invGamma = 1.0 / gamma
        	table = np.array([(((i / 255.0)) ** invGamma) * 255
        		for i in np.arange(0, 256)]).astype("uint8")
        	# apply gamma correction using the lookup table
        	return cv2.LUT(image, table)
        
        # ----------------------------------------------------------------------------
        # Preprocessing image
        # ----------------------------------------------------------------------------
        
        gray = adjust_gamma(img,2.5)
        blur = cv2.GaussianBlur(gray, (11,11), 0)
        thresh2= cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,17,2)
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        close = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
        close = close - 255
        
        
        # Erase small areas
        cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        result = np.zeros(img.shape, np.uint8)
        
        for c in cnts:  
            area = cv2.contourArea(c)

            if area > 500:
                cv2.drawContours(result, [c], -1, (255,0,0), -1)

        if visible == True:
            Images =[img,gray,blur,thresh2,close,result]
            titles =['Band'+str(band),'gray','blur','thresh2','close','result']
            n=len(Images)
            for i in range(n):
                plt.subplot(n/2,2,i+1),plt.imshow(Images[i],'gray')
                plt.title(titles[i])
                plt.xticks([]),plt.yticks([])
            plt.show()
        # ----------------------------------------------------------------------------
        # matching con template  
        # ----------------------------------------------------------------------------
        
        template = cv2.imread(patternname,0)
        
        template = cv2.Canny(template, 50, 200)

        (tH, tW) = template.shape[:2]
        if visible == True:
            Images =[result, template]
            titles =['result','template']
            n=len(Images)
            for i in range(n):
                plt.subplot(n/2,2,i+1),plt.imshow(Images[i],'gray')
                plt.title(titles[i])
                plt.xticks([]),plt.yticks([])
            plt.show()
    
        found = None
        gray = result
        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
          		# resize the image according to the scale, and keep track
          		# of the ratio of the resizing
          		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
          		r = gray.shape[1] / float(resized.shape[1])
          		# if the resized image is smaller than the template, then break
          		# from the loop
          		if resized.shape[0] < tH or resized.shape[1] < tW:
          			break
          		# detect edges in the resized, grayscale image and apply template
          		# matching to find the template in the image
          		edged = cv2.Canny(resized, 50, 200)
          		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
          		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                  
          		if found is None or maxVal > found[0]:
          			found = (maxVal, maxLoc, r)
   
                       
        founds.append(found)          
    
               
    coef = [row[0] for row in founds]           
    ind = np.argmax(coef)           
    (_, maxLoc, r) = founds[ind]
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    k=2
    i=0
    msi=[]
    for band in range (0,N):
        img = images[band]
        (m,n)=img.shape
        if startY-k < 1 or endY+k > m or startX-k < 1 or endX+k > n:
            print('Fuera de lim')
            ROI = img[startY:endY,startX:endX]
        else:
            ROI = img[startY-k:endY+k,startX-k:endX+k]
        msi.append(ROI)
        if visible == True:
            plt.subplot(nf,nc,i+1),plt.imshow(ROI,'gray'),plt.title(Titles[i])
            plt.xticks([]),plt.yticks([])
            i=i+1
    if visible == True:          
        plt.subplots_adjust(hspace=1, wspace=0)    
        plt.show()
        plt.close()
        cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 2)
        plt.imshow(img,'gray')
        
    return msi

