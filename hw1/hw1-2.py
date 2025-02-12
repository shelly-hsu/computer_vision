import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def gaussian_blur(img, ksize, sigma):
   
    # generate 1D Gaussian kernel
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    # get 2D Gaussian kernel by multiplying two 1D Gaussian kernel
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    # normalized Gaussian kernel
    norm_kernel = kernel_2d / kernel_2d.sum()
    # apply normalized 2D Gaussian kenrel to image 
    filtered_img = cv2.filter2D(img, -1, norm_kernel)

    return filtered_img

def sobel_x(img_blur):
    # create Sobel filter to detect horizontal edges
    filter = np.array([[-1,0,1], 
                      [-2,0,2],
                      [-1,0,1]])
    # apply filter to the ouput image of Gaussain_blur function
    gradient_x = cv2.filter2D(img_blur, cv2.CV_64F, filter)
    return gradient_x  

def sobel_y(img_blur):
    # create Sobel filter to detect vertical edges
    filter = np.array([[-1,-2,-1], 
                      [0,0,0],
                      [1,2,1]])
    # apply filter to the ouput image of Gaussain_blur function
    gradient_y = cv2.filter2D(img_blur, cv2.CV_64F, filter)
    return gradient_y

def corner_response(dx, dy, ksize, threshold):
    
    print('corner response...')
    k= 0.04
    height, width = dx.shape
    offset = ksize // 2

    # Caculate the structure matrix element
    dxx = gaussian_blur(dx * dx, ksize, 3)
    dyy = gaussian_blur(dy * dy, ksize, 3)
    dxy = gaussian_blur(dx * dy, ksize, 3)

    # calculate the harris response matrix R
    det = dxx * dyy - dxy ** 2  # determinant
    tr = dxx + dyy  # trace
    R = det - k * tr ** 2   # harris response matrix R

    img_corner = np.zeros_like(dx)
    #recaculate threshild because the harris response matrix is not normalized
    threshold = threshold * R.max()
    # thresholding : loop through the harris response matrix to check which position is larger than threshold
    for y in range (offset, height-offset):
        for x in range(offset, width-offset):
            # if it is large than threshold, then set the value to 255(white) of the same position in img_corner array
            if R[y, x] > threshold:
                img_corner[y, x] = 255
        
    return R, img_corner


def nms(R, nms_window, threshold):

    print('nms...')
    height, width = R.shape
    offset = nms_window // 2
    img_nms = np.zeros_like(R)
    threshold = threshold * R.max()
    nms_pos = []
    # loop through the harris response matrix with a 5x5 window
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # local_max is the maximum number in the 5x5 window
            local_max = np.max(R[y - offset:y + offset + 1, x-offset : x+offset+1])
            # check whather the (y,x) positon is the local_max and larger than the threshld or not
            if R[y, x] == local_max and R[y, x] > threshold:
                # if the condition holds, then set the corresponding poisiton in img_mns array to 255(white)
                img_nms[y,x] = 255      
                nms_pos.append((y,x))   # record the nms coordinate
    return nms_pos, img_nms

def combine(img, nms_pos):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for corner in nms_pos:
        x, y = corner
        cv2.circle(rgb_image, (y, x), 3, (255, 0, 0), -1)
    return rgb_image

if __name__ == '__main__':
    
    sigma = 3
    img = cv2.imread('hw1-2.jpg', cv2.IMREAD_GRAYSCALE)

    # Original settting with kernel_size=3x3, threshold=0.01, nms_window_size=3x3
    ksize = 3
    threshold = 0.01
    img_blur = gaussian_blur(img, ksize, sigma)
    gradient_x = sobel_x(img_blur)
    gradient_y = sobel_y(img_blur)
    R, img_corner = corner_response(gradient_x, gradient_y, ksize, threshold)
    nms_pos, img_nms = nms(R, 5, threshold)
    img_combine = combine(img, nms_pos)
    print(len(nms_pos))
   
    ########################################################################################
    # Addition seeting 1) with kernel_size=5x5, threshold=0.01, nms_window_size=3x3
    print('Additional setting ksize=5...')
    ksize=5
    img_blur_1 = gaussian_blur(img, 3,sigma)
    gradient_x_1 = sobel_x(img_blur)
    gradient_y_1 = sobel_y(img_blur)
    R, img_corner_1 = corner_response(gradient_x, gradient_y, ksize, threshold)
    
    ########################################################################################
    # Addition setting 2) with kernel_size=3x3, thresold=0.03, nms_window_size=5x5
    print('Additional setting threshold=0.03...')
    ksize=3
    threshold=0.03
    img_blur_2 = gaussian_blur(img, ksize,sigma)
    gradient_x_2 = sobel_x(img_blur)
    gradient_y_2 = sobel_y(img_blur)
    R, img_corner_2 = corner_response(gradient_x, gradient_y, ksize, threshold)

    f1, ax = plt.subplots(2,3, figsize=(12,6))
    ax[0][0].imshow(img_blur, cmap='gray')
    ax[0][1].imshow(gradient_x, cmap='gray')
    ax[0][2].imshow(gradient_y, cmap='gray')
    ax[1][0].imshow(img_corner, cmap='gray')
    ax[1][1].imshow(img_nms, cmap='gray')
    ax[1][2].imshow(img_combine)
    
    ax[0][0].axis('off')
    ax[0][1].axis('off')
    ax[0][2].axis('off')
    ax[1][0].axis('off')
    ax[1][1].axis('off')
    ax[1][2].axis('off')
    ax[0][0].title.set_text('Gaussian_blur')
    ax[0][1].title.set_text('Sobel_x')
    ax[0][2].title.set_text('Sobel_y')
    ax[1][0].title.set_text('Corner_response')
    ax[1][1].title.set_text('Non_maximal_suppression')
    ax[1][2].title.set_text('Combine')
    
    f2, ax2 = plt.subplots(1,3, figsize=(12,6))
    ax2[0].imshow(img_corner, cmap='gray')
    ax2[1].imshow(img_corner_1, cmap='gray')
    ax2[2].imshow(img_corner_2, cmap='gray')
    
    ax2[0].axis('off')
    ax2[1].axis('off')
    ax2[2].axis('off')
    ax2[0].title.set_text('original')
    ax2[1].title.set_text('ksize=5')
    ax2[2].title.set_text('threshold=0.03')
    
    plt.figure(3)
    plt.imshow(img_blur, cmap='gray')
    plt.axis('off')
    plt.title('Result after Gaussian blur')

    plt.figure(4)
    plt.imshow(gradient_x, cmap='gray')
    plt.axis('off')
    plt.title('Result after Sobel operation(x direction)')

    plt.figure(5)
    plt.imshow(gradient_y, cmap='gray')
    plt.axis('off')
    plt.title('Result after Sobel operation(y direction)')

    plt.figure(6)
    plt.imshow(img_corner, cmap='gray')
    plt.axis('off')
    plt.title('Harris response')

    plt.figure(7)
    plt.imshow(img_nms, cmap='gray')
    plt.axis('off')
    plt.title('Result after non-maximal suppression')

    plt.figure(8)
    plt.imshow(img_combine)
    plt.axis('off')
    plt.title('Final output')

    plt.figure(9)
    plt.imshow(img_corner_1, cmap='gray')
    plt.axis('off')
    plt.title('ksize=5')

    plt.figure(10)
    plt.imshow(img_corner_2, cmap='gray')
    plt.axis('off')
    plt.title('threshold=0.03')
    
    destination = './result_image/hw1-2(b.)(i).jpg'
    f1.savefig(destination)
    cv2.imwrite('./result_image/hw1-2(a)(i).jpg', img_blur)
    plt.imsave('./result_image/hw1-2(a)(ii)_x_direction.jpg', gradient_x, cmap='gray')
    plt.imsave('./result_image/hw1-2(a)(ii)_y_direction.jpg', gradient_y, cmap='gray')
    cv2.imwrite('./result_image/hw1-2(a)(iii).jpg', img_corner)
    cv2.imwrite('./result_image/hw1-2(a)(iv).jpg', img_nms)
    plt.imsave('./result_image/hw1-2(b)(ii).jpg', img_combine)
    cv2.imwrite('./result_image/hw1-2(c.)(i).jpg', img_corner_1)
    cv2.imwrite('result_image/hw1-2(c.)(ii).jpg', img_corner_2)
    plt.show()