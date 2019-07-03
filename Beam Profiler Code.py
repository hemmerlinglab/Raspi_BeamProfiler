#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy import optimize
import scipy


# basic array handling

def array_norm(n): #normalizes an array (sets all values inbetween 0 and 1) ***
    x = n - np.min(n)
    x = n/np.max(n)
    return(x)

def array_range(n): #gives the minimum and maximum values in an array
    return([np.min(n),np.max(n)])

def proj_norm(n): #projects an array onto the x and y axis and normalizes the values ***
    xNorm = np.sum(n,0)
    xNorm = xNorm/np.amax(xNorm)
    yNorm = np.sum(n,1)
    yNorm = yNorm/np.amax(yNorm)
    return(xNorm,yNorm)


# functions to display and process images

def get_img(file): #converts an image from the home folder to a normalized grayscale image
    x = np.array(Image.open(file).convert("L"))
    x = array_norm(x)
    x = np.around(x, decimals=4)
    return(x)

def blur_img(img, lvl): #blurs an image with the level of blurring determined by lvl (~10:light ~100:heavy)
    x = scipy.ndimage.gaussian_filter(img,lvl,order=0,truncate=1.0)
    return(x)

def clip_img(n, limit): #sets values in an array that are below the limit to zero
    x = np.clip(array_norm(n),limit,1) - np.clip(array_norm(n),limit,0)
    return(x)

def show_img(n): #shortcut to display an image
    plt.axis('off')
    plt.imshow(n);plt.show()
    
def output_proj(n): #displays an image and its' projections
    x = proj_norm(n)
    print("x-projection:")
    plt.plot(x[0]);plt.show()
    print("y-projection:")
    plt.plot(x[1]);plt.show()

def img_center(img): #gives the pixel analogous to the image's center of mass ***
    img = proj_norm(img)
    (x, y) = (img[0], img[1]) 
    xC = np.sum(x*range(0,len(x)))/np.sum(x)
    yC = np.sum(y*range(0,len(y)))/np.sum(y)
    return(np.round(xC,0), np.round(yC,0))

def img_focus(n, xDim, yDim): #gives an image cropped around the image's "center" with dimensions:(xDim,yDim)
    mid = img_center(n)
    x = int(mid[0]); y = int(mid[1])
    xDim = xDim//2; yDim = yDim//2
    return(n[y-yDim:y+yDim,x-xDim:x+xDim])



# function fitting for arrays and images

def gauss(x, a, b, c): # a basic form of the gaussian distribution ***
    return((np.exp(-2*np.power(((x - a)/b),2))*c))

def fit_func(func, data, par): #fits a function to the data with initial values par ***
    fit = scipy.optimize.curve_fit(func, range(0,len(data)), data, p0=par)[0]
    return(fit)

def gauss_fit(data): #performs a gaussian fit for a normalized 1D array
    fit = fit_func(gauss,data,[np.argmax(data),200,1.1])
    x = np.array(range(0,len(data)))
    y = gauss(x,fit[0],fit[1],fit[2])
    y = array_norm(y) 
    return(fit,x,y)


def waist_display(img): #displays the waist and a fit for the x and y projections of image
    x = proj_norm(img)[0]; xFit = gauss_fit(x)
    y = proj_norm(img)[1]; yFit = gauss_fit(y)
    print("x-radius:", beam_waist(img)[0], "um")
    plt.plot(x); plt.plot(xFit[1],xFit[2]); plt.show()
    print("y-radius:", beam_waist(img)[1], "um")
    plt.plot(y); plt.plot(yFit[1],yFit[2]); plt.show()
    

#pixels on the rasberry pi (V.2) camera are 1.12 x 1.12 micrometers

def beam_waist(img): #finds an img's beam waist from a gaussian fit
    proj = proj_norm(img)
    xWaist = (gauss_fit(proj[0])[0])[1]*1.12
    xWaist = np.round(xWaist,1)
    yWaist = (gauss_fit(proj[1])[0])[1]*1.12
    yWaist = np.round(yWaist,1)
    return(xWaist,yWaist)

def beam_waist1(data): #gives the beam waist of a data set based of a 1/e^2 cutoff
    waist = np.where(gauss_fit(data)[2]>.135)
    waist = np.argmax(waist)-np.argmin(waist)
    waist = np.round(waist*0.66,1)
    return(waist)


# stuff for handling over-saturated images

def trim_array(data, floor, limit): #deletes values from a data set above the limit or below the floor
    index = range(0,len(data))
    data = np.dstack((index, data))[0]
    
    a = np.array([])
    for n in index:
        if (data[n])[1] < floor or (data[n])[1] > limit:
            a = np.append(a, n)
    a = a.astype(int)
    
    out = np.delete(data,a,0)
    x = out.flatten()[0::2]
    y = out.flatten()[1::2]
    return(x,y)

def saturated_fit(data, a, b): #performs a gaussian fit for a saturated beam ***
    trim = trim_array(data, a, b)
    x = trim[0]; y = trim[1];
    fit = scipy.optimize.curve_fit(gauss, x, y, p0=[np.argmax(data),100,1])[0]
    new = gauss(range(len(data)),fit[0],fit[1],fit[2])
    return(new, np.around(fit[1]*1.12,1))

def saturated_display(img, floor, limit): #displays the waists and fits for saturated images
    x = proj_norm(img)[0]; xFit = saturated_fit(x,floor,limit)
    y = proj_norm(img)[1]; yFit = saturated_fit(y,floor,limit)
    
    print("x-radius:", xFit[1], "um")
    plt.plot(x); plt.plot(xFit[0]); plt.show()
    print("y-radius:", yFit[1], "um")
    plt.plot(y); plt.plot(yFit[0]); plt.show()
    

#look into D4sigma beam width
    
# image definitions
cat   = get_img('cat.png')
beam1 = get_img('beam1.png')
ideal = get_img('idealBeam.png')
direct2 = get_img('direct2.png')
image27 = get_img('image27.png')
dist38 = get_img('dist38mm.png')
dist63 = get_img('dist63mm.png')
dist89 = get_img('dist89mm.png')
dist114 = get_img('dist114mm.png')
dist140 = get_img('dist140mm.png')
dist165 = get_img('dist165mm.png')


# In[63]:


test = dist114
test = img_focus(test,800,800)
test = img_focus(test,500,500)


show_img(test)

saturated_display(test,0,0.5)

#waist_display(test)
#data = proj_norm(test)[0]
#print(np.diff(array_range(np.where(data > np.mean(data)+np.std(data))))[0])


# In[ ]:




