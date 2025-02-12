import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalizeHistogram(img):

    # caculate histogram
    histogram, bins = np.histogram(img.flatten(), bins=256, range=[0,256])

    # caculate cumulative histogram
    cdf = np.cumsum(histogram)

    # caculate normalized cumulative histogram
    cdf_norm = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    #apply cumulative distribution function transformation
    img_eq = cdf_norm[img].astype(np.uint8)

    return histogram, img_eq
    
if __name__ == '__main__':

   
    img = cv2.imread('hw1-1.jpg',cv2.IMREAD_GRAYSCALE)
    histogram, img_eq = equalizeHistogram(img)
    
    f, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(img_eq, cmap='gray')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].title.set_text('Original')
    ax[1].title.set_text('Equalized')
    

    f2, ax2 = plt.subplots(1,2, figsize=(10,5))
    ax2[0].hist(img.flatten(),bins=256, range=[0,256], alpha=0.5)
    ax2[1].hist(img_eq.flatten(),bins=256, range=[0,256], alpha=0.5)
    ax2[0].title.set_text('Original histogram')
    ax2[1].title.set_text('Equalized histogram')

    destination = './result_image/hw1-1(b).jpg'
    f.savefig(destination)
    destination = './result_image/hw1-1(c).jpg'
    f2.savefig(destination)
    
    plt.show()
    