#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


from PIL import Image, ImageFilter
img = Image.open(r'C:\Users\PC\Pictures\Video Projects\dream.jpg') 


# In[3]:


from skimage.restoration import denoise_tv_chambolle


# In[4]:


def plot_comparison(img_original, img_filtered, img_title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    ax1.imshow(img_original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(img_filtered, cmap=plt.cm.gray)
    ax2.set_title(img_title_filtered)
    ax2.axis('off')


# In[5]:


enc_img = img.filter(ImageFilter.DETAIL)
plot_comparison(img, enc_img, 'detail image')


# In[6]:


from skimage.util import random_noise
noisy_image = plt.imread(r'C:\Users\PC\Pictures\Video Projects\dream.jpg')
fruit_image = plt.imread(r'C:\Users\PC\Pictures\Video Projects\dream.jpg')

# Add noise to the image
noisy_image = random_noise(fruit_image)

# Show th original and resulting image
plot_comparison(fruit_image, noisy_image, 'Noisy image')


# In[7]:


from skimage.segmentation import slic
from skimage.color import label2rgb

face_image = plt.imread(r'C:\Users\PC\Pictures\Video Projects\dream.jpg')

# Obtain the segmentation with 400 regions
segments = slic(face_image, n_segments=400)

# Put segments on top of original image to compare
segmented_image = label2rgb(segments, face_image, kind='avg')

# Show the segmented image
plot_comparison(face_image, segmented_image, 'Segmented image, 400 superpixels')


# In[8]:


def show_image_contour(image, contours):
    plt.figure()
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3)
    plt.imshow(image, interpolation='nearest', cmap='gray_r')
    plt.title('Contours')
    plt.axis('off')


# In[9]:


from skimage.restoration import denoise_tv_chambolle

noisy_image = plt.imread(r'C:\Users\PC\Pictures\Video Projects\dream.jpg')

# Apply total variation filter denoising
denoised_image = denoise_tv_chambolle(noisy_image, multichannel=True)

# Show the noisy and denoised image
plot_comparison(noisy_image, denoised_image, 'Denoised Image')


# In[ ]:




