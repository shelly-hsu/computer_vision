import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import time

def readFile(file_path):
    data = []  
    with open(file_path, "r") as file:
        for i, line in enumerate(file, start=1):
            if i >= 2:
                x, y = map(float, line.strip().split())
                data.append((x, y, 1))  
    data = np.array(data)
    return data

def normalization_matrix(points):

    mean = np.mean(points, axis=0)  # Calculate the mean along column respresent the mean of x, y 
    std = np.std(points)

    # Calculate scale factor
    s = np.sqrt(2) / std

    # Create the normalization matrix
    T = np.array([[s, 0, -s * mean[0]],
                  [0, s, -s * mean[1]],
                  [0, 0, 1]])
    norm_points = (T @ points.T).T

    return T, norm_points

def eight_point_algo(points1, points2):
    
    A = np.zeros((59, 9))

    for i in range(59):
        x1, y1, _ = points1[i]
        x2, y2, _ = points2[i]
        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]

    U, D, Vt = np.linalg.svd(A)   # perform SVD on matrix A
    F = Vt[-1].reshape(3, 3)    # Extract the fundamental matrix F from the last column of V
    
    # Enforce the rank-2 constrain on F
    U_F, D_F, Vt_F = np.linalg.svd(F)
    D_F[-1] = 0     # set the last column in D into zeros
    F = U_F @ np.diag(D_F) @ Vt_F
    
    return F

def norm_eight_point_algo(points1, points2):
    
    A = np.zeros((59, 9))
    T1, norm_points1 = normalization_matrix(points1)
    T2, norm_points2 = normalization_matrix(points2)
    
    for i in range(59):
        x1, y1, _ = norm_points1[i]
        x2, y2, _ = norm_points2[i]
        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]

    U, D, Vt = np.linalg.svd(A)   # perform SVD on matrix A
    F = Vt[-1].reshape(3, 3)    # Extract the fundamental matrix F from the last column of V
    
    # Enforce the rank-2 constrain on F
    U_F, D_F, Vt_F = np.linalg.svd(F)
    D_F[-1] = 0     # set the last column in D into zeros
    F = U_F @ np.diag(D_F) @ Vt_F

    F = T2.T @ F @ T1

    # F /= F[2, 2]       # Normalize the fundamental matrix by its last element
    # print(F)
    return F

def interpolate_color(index, total_lines):
    # Interpolate color from red to purple to blue based on the index
    red = int(255 * (1 - index / total_lines))
    blue = int(255 * (index / total_lines))
    return (blue, 0, red)



def plot_epipolar_lines(F, img1, img2, points1, points2):
    
    # Create a canvas for drawing
    canvas1 = np.copy(img1)
    canvas2 = np.copy(img2)

    # Create a list of line segments to be drawn
    lines1 = []
    lines2 = []
    n = len(points1)
    distances1 = np.zeros(n)
    distances2 = np.zeros(n)

    for i in range(len(points1)):
        color = interpolate_color(i, n)

        point1 = points1[i]
        point2 = points2[i]
        # Compute the epipolar line in both images
        epipolar_line1 = (F @ point2.T).T
        epipolar_line2 = (F.T @ point1.T).T  # Transpose F for the second image
        a1, b1, c1 = epipolar_line1
        a2, b2, c2 = epipolar_line2

        x1 = 0
        x2 = img1.shape[1] - 1
        y1 = int((-c1 - a1 * x1) / b1)
        y2 = int((-c1 - a1 * x2) / b1)
        cv2.line(canvas1, (x1, y1), (x2, y2), color, 1)
        distance = abs(a1 * point2[0] + b1 * point2[1] + c1) / np.sqrt(a1**2 + b1**2)
        distances1[i] = distance

        x1 = 0
        x2 = img2.shape[1] - 1
        y1 = int((-c2 - a2 * x1) / b2)
        y2 = int((-c2 - a2 * x2) / b2)
        cv2.line(canvas2, (x1, y1), (x2, y2), color, 1)
        distance = abs(a2 * point2[0] + b2 * point2[1] + c2) / np.sqrt(a2**2 + b2**2)
        distances2[i] = distance

    average_distance1 = np.mean(distances1)
    average_distance2 = np.mean(distances2)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.cvtColor(canvas2, cv2.COLOR_BGR2RGB))  # Convert to RGB for display
    ax[1].imshow(cv2.cvtColor(canvas1, cv2.COLOR_BGR2RGB))
    ax[0].axis('off')
    ax[1].axis('off')
    plt.subplots_adjust(wspace=0) 

    return average_distance1, average_distance2

if __name__ == '__main__':
    

    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')
    points1 = readFile(file_path = 'pt_2D_1.txt')
    points2 = readFile(file_path = 'pt_2D_2.txt')
   
    F = eight_point_algo(points1, points2)
    F_norm = norm_eight_point_algo(points1, points2)

    average_distance1, average_distance2 = plot_epipolar_lines(F, img1, img2, points1, points2)
    average_distance1_norm, average_distance2_norm = plot_epipolar_lines(F_norm, img1, img2, points1, points2)
    print(average_distance1, average_distance2)
    print(average_distance1_norm, average_distance2_norm)
    plt.show()