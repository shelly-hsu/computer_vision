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
    return T

def eight_point_algo(points_1, points_2):
    
    A = np.zeros((8, 9))

    for i in range(8):
        x1, y1, _ = points_1[i]
        x2, y2, _ = points_2[i]
        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]
    U, D, Vt = np.linalg.svd(A)   # perform SVD on matrix A
    F = Vt[-1].reshape(3, 3)    # Extract the fundamental matrix F from the last column of V
    
    # Enforce the rank-2 constrain on F
    U_F, D_F, Vt_F = np.linalg.svd(F)
    D_F[-1] = 0     # set the last column in D into zeros
    F = U_F @ np.diag(D_F) @ Vt_F
    
    F /= F[2, 2]       # Normalize the fundamental matrix by its last element
    return F

def compute_best_fundamental_matrix(points1, points2):
    
    print('finding')
    random_seed = 6  # You can use any integer as the seed
    np.random.seed(random_seed)
    
    iterations = 1
    threshold = 0.01
    best_F = None
    best_inlier_count = -1

    for i in range(iterations):

        # Randomly sample 8 points for the Eight-Point Algorithm
        random_indices = np.random.choice(len(points1), size=8, replace=False)
        sampled_points1 = points1[random_indices]
        sampled_points2 = points2[random_indices]

        # Use the Eight-Point Algorithm to compute the fundamental matrix
        F = eight_point_algo(sampled_points1, sampled_points2)

       
        # points_2 @ F @ points_1.T # computes the epipolar constraints for each corresponding pair of points, so it is expected to be close to zero
        # Count inliers
       # Calculate the epipolar lines and count inliers
        epipolar_lines = points2 @ F @ points1.T
        residuals = np.abs(epipolar_lines)
        inliers = np.where(residuals < threshold)[0]
        inlier_count = len(inliers)
    
        # If this is the best model so far, update the best model
        if inlier_count > best_inlier_count:
            best_F = F
            best_inlier_count = inlier_count
    return best_F

def compute_best_norm_fundamental_matrix(points1, points2):
    
    iterations = 10000
    threshold = 0.01
    best_F = None
    best_inlier_count = -100

    for i in range(iterations):
        # Randomly sample 8 points for the Eight-Point Algorithm
        random_indices = np.random.choice(len(points1), size=8, replace=False)
        sampled_points1 = points1[random_indices]
        sampled_points2 = points2[random_indices]

        # normalization
        T1 = normalization_matrix(sampled_points1)
        T2 = normalization_matrix(sampled_points2)
        sampled_points1 = (T1 @ sampled_points1.T).T
        sampled_points2 = (T2 @ sampled_points2.T).T

        # Use the Eight-Point Algorithm to compute the fundamental matrix
        F = eight_point_algo(sampled_points1, sampled_points2)
        # denormalized 
        F = T2.T @ F @ T1
       
        # points_2 @ F @ points_1.T # computes the epipolar constraints for each corresponding pair of points, so it is expected to be close to zero
        # Count inliers
        errors = np.sum(points2 @ F @ points1.T, axis=1) # calculate the sum along the rows which represent t
       
        inliers = np.where(abs(errors) < threshold)[0]
      
        inlier_count = len(inliers)

        # If this is the best model so far, update the best model
        if inlier_count > best_inlier_count:
            best_F = F
            best_inlier_count = inlier_count
    if inlier_count >= 8: 
        inlier_points1 = points1[inliers]
        inlier_points2 = points2[inliers]
        best_F = eight_point_algo(inlier_points1, inlier_points2)
    return best_F

def calculate_epipolar_lines(F, points):
  
    # Compute the epipolar lines
    epipolar_lines = (F @ points.T).T
    return epipolar_lines

# def plot_epipolar_lines(img, points, F):
    
#     epipolar_lines = calculate_epipolar_lines(F, points)
  
    
#     for line in epipolar_lines:
#         a, b, c = line
#         y = np.arange(0, img.shape[0])
#         x = (-b * y - c) / a

#         # Filter points that are within the image bounds
#         valid_indices = (x >= 0) & (x < img.shape[1])
#         x, y = x[valid_indices], y[valid_indices]

#         for i in range(len(x)):
#             cv2.circle(img, (int(x[i]), int(y[i])), 1, (0, 0, 255), -1)

#     plt.imshow(img)
#     plt.show()

def plot_epipolar_lines(F, img1, img2, points1, points2):
    

    # Create a canvas for drawing
    canvas = np.copy(img2)

    # Create a list of line segments to be drawn
    lines = []
    for i in range(len(points1)):
        point1 = points1[i]
        point2 = points2[i]
        # Compute the epipolar line in the second image
        epipolar_line = (F @ point1.T).T
        a, b, c = epipolar_line

        x1 = 0
        x2 = img2.shape[1] - 1
        y1 = int((-c - a * x1) / b)
        y2 = int((-c - a * x2) / b)

        lines.append(np.array([(x1, y1), (x2, y2)], dtype=np.int32))

    # Draw all the lines on the canvas
    cv2.polylines(canvas, lines, isClosed=False, color=(0, 255, 0), thickness=1)

    # Show the image with epipolar lines
    cv2.imshow("Image with Epipolar Lines", canvas)
    cv2.waitKey(0)

    # plt.figure(figsize=(15, 5))
    
    # for i in range(len(points1)):
    #     point1 = points1[i]
    #     point2 = points2[i]
    #     # Compute the epipolar line in the second image
    #     epipolar_line = (F @ point1.T).T

    #     # Extract coefficients of the line (a, b, c)
    #     a, b, c = epipolar_line

    #     x1 = 0
    #     x2 = img2.shape[1] - 1  # Width of the image

    #     # Calculate the corresponding y-coordinates using the equation of the line
    #     y1 = int((-c - a * x1) / b)
    #     y2 = int((-c - a * x2) / b)

    #     color = (0, 255, 0)  # RGB color (green in this example)
    #     thickness = 1  # Line thickness

    #     # Draw the line on the image
    #     cv2.line(img2, (x1, y1), (x2, y2), color, thickness)

    # cv2.imshow("Image with Epipolar Line", img2)
    # cv2.waitKey(0)

if __name__ == '__main__':
    
    start_time = time.time()
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')
    points1 = readFile(file_path = 'pt_2D_1.txt')
    points2 = readFile(file_path = 'pt_2D_2.txt')
    print(img1.shape)
    F = compute_best_fundamental_matrix(points1, points2)
    # F = eight_point_algo(points1, points2)
    # print(F)
    F_norm = compute_best_norm_fundamental_matrix(points1, points2)
    end_time = time.time()
    print(end_time - start_time)
    plot_epipolar_lines(F, img2, img1, points2, points1)
    
    


    # def plot_epipolar_lines(img, points, F):
    
#     epipolar_lines = calculate_epipolar_lines(F, points)
  
    
#     for line in epipolar_lines:
#         a, b, c = line
#         y = np.arange(0, img.shape[0])
#         x = (-b * y - c) / a

#         # Filter points that are within the image bounds
#         valid_indices = (x >= 0) & (x < img.shape[1])
#         x, y = x[valid_indices], y[valid_indices]

#         for i in range(len(x)):
#             cv2.circle(img, (int(x[i]), int(y[i])), 1, (0, 0, 255), -1)

#     plt.imshow(img)
#     plt.show()
