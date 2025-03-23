import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


def sift_extractor(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    # img_keypoints = cv2.drawKeypoints(img, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return keypoints, descriptors

def feature_matching(desc_source, desc_target, ratio_threshold):

    print('matching...')
    matches=[]
    
   
    for i, desc1 in enumerate(desc_source):
       
        best_match = None

        # find the distance
        distances = np.linalg.norm(desc1 - desc_target, axis=1)
            
        # ratio test
        indices = list(range(len(distances)))
        sorted_indices = sorted(indices, key=lambda i: distances[i])
        best_distance = distances[sorted_indices[0]]
        second_best_distance = distances[sorted_indices[1]]
        if best_distance < ratio_threshold * second_best_distance:
           best_match = cv2.DMatch(i, sorted_indices[0], best_distance)

    
        if best_match is not None:
           matches.append(best_match)

    return matches

def draw_line(fig, des_points):
    des_points = des_points[:, :2]
    for i in range(len(des_points)):
        # Draw lines between adjacent points to form a quadrilateral
        pt1 = tuple(map(int, des_points[i]))
        pt2 = tuple(map(int, des_points[(i + 1) % 4]))
        cv2.line(fig, pt1, pt2, (255, 0, 0), 5)
    return fig

def draw_deviation(img, H, src_point, dst_point, color):
    # Transform source points using the given homography matrix
    trans_point = np.dot(src_point, H.T)
    trans_point = trans_point / trans_point[:, [-1]]  # Normalize homogeneous coordinates
    trans_point = trans_point[:, :2]  
    dst_point = dst_point[:, :2]  

    # Calculate the deviation vector between transformed points and destination points
    deviation_vector = trans_point - dst_point
    print(deviation_vector) 

    # Draw arrows to represent deviation vectors for each corresponding point pair
    for i, (point, deviation_vector) in enumerate(zip(dst_point, deviation_vector)):
        point_start = (int(point[0]), int(point[1]))  # Starting point of the arrow
        point_end = (int(point[0] + deviation_vector[0]), int(point[1] + deviation_vector[1]))  
        cv2.arrowedLine(img, point_start, point_end, color, thickness=10, tipLength=0.3)  

    return img  

def get_matching_point(matches, keypoint_source, keypoint_target):
    src_pts = []
    dst_pts = []
    for match in matches:
        src_idx = match.queryIdx
        dst_idx = match.trainIdx
        src_pts.append(keypoint_source[src_idx].pt)
        dst_pts.append(keypoint_target[dst_idx].pt)

    src_pts = np.array(src_pts).astype(np.float32)
    dst_pts = np.array(dst_pts).astype(np.float32)
    return src_pts, dst_pts

def homography(src_sampled, dst_sampled):
    A = []
    src_sampled = src_sampled[:,:2]
    dst_sampled = dst_sampled[:,:2]
    for j in range(4):
        x, y = src_sampled[j][0], src_sampled[j][1]
        u, v = dst_sampled[j][0], dst_sampled[j][1]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    A = np.array(A)
    U, D, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape(3, 3)
    H = H / H[2, 2]
    return H

def get_homography_ransac(src_pts, dst_pts, num_iterations, threshold):
    
    print('ransac...')
    best_H = None
    best_mask = None
    best_inliers = 0
    np.random.seed(3)

    for i in range(num_iterations):
        # Randomly sample 4 points
        indices = np.random.choice(len(src_pts), 4, replace=False)
        src_sampled = src_pts[indices]
        dst_sampled = dst_pts[indices]
       
        # compute the homograpghy matrix according to hw2-2 code
        H = homography(src_sampled, dst_sampled)
        # Apply homography to source points
        transformed_src = np.dot(src_pts, H.T)
        transformed_src = transformed_src / transformed_src[:, [-1]]

        # Calculate distance between transformed points and destination points
        distances = np.linalg.norm(transformed_src - dst_pts, axis=1)
        # print(distances)
        # Count inliers based on the threshold
        inliers = np.sum(distances < threshold)
        # Keep track of the best homography with most inliers
        mask = (distances < threshold).astype(np.uint8)
        # Keep track of the best homography with most inliers
        if inliers > best_inliers:
            best_inliers = inliers
            best_mask = mask
            best_H = H
    print('H=', best_H)
    return best_H, best_mask

def good_matches(matches):
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Adjust the threshold (e.g., 0.75) as needed
            good_matches.append(m)
    return good_matches


if __name__ == '__main__':
    
    start_time = time.time()
    img_target = cv2.imread('1-image.jpg')
    img_book_1 = cv2.imread('1-book1.jpg')
    img_book_2 = cv2.imread('1-book2.jpg')
    img_book_3 = cv2.imread('1-book3.jpg')

    point_1 = np.array([(98,220,1), (1002,223,1), (995,1330,1), (124,1348,1)]).astype(np.float32)
    point_2 = np.array([(66,121,1), (1042,113,1), (1040,1340,1), (74,1348,1)]).astype(np.float32)
    point_3 = np.array([(127,185,1), (992,180,1), (981,1394,1),(135, 1394,1)]).astype(np.float32)

    keypoint_target, desc_target = sift_extractor(img_target)
    keypoint_book_1, desc_book_1 = sift_extractor(img_book_1)
    keypoint_book_2, desc_book_2 = sift_extractor(img_book_2)
    keypoint_book_3, desc_book_3 = sift_extractor(img_book_3)
    
    # bf = cv2.BFMatcher()
    # matches_book_1 = bf.knnMatch(desc_book_1, desc_target, k=2)
    # matches_book_1 = good_matches(matches_book_1)
    # matches_book_2 = bf.knnMatch(desc_book_2, desc_target, k=2)
    # matches_book_2 = good_matches(matches_book_2)
    # matches_book_3 = bf.knnMatch(desc_book_3, desc_target, k=2)
    # matches_book_3 = good_matches(matches_book_3)

    # find matching
    matches_book_1 = feature_matching(desc_book_1, desc_target, 0.75)
    matches_book_2 = feature_matching(desc_book_2, desc_target, 0.75)
    matches_book_3 = feature_matching(desc_book_3, desc_target, 0.75)

    src_pts_1, dst_pts_1 = get_matching_point(matches_book_1, keypoint_book_1, keypoint_target)
    src_pts_2, dst_pts_2 = get_matching_point(matches_book_2, keypoint_book_2, keypoint_target)
    src_pts_3, dst_pts_3 = get_matching_point(matches_book_3, keypoint_book_3, keypoint_target)

    matches_1 = [[match] for match in matches_book_1]  
    matches_2 = [[match] for match in matches_book_2] 
    matches_3 = [[match] for match in matches_book_3]

    ones_column = np.ones((src_pts_1.shape[0], 1))
    src_pts_1 = np.hstack((src_pts_1, ones_column))
    dst_pts_1 = np.hstack((dst_pts_1, ones_column))
    ones_column = np.ones((src_pts_2.shape[0], 1))
    src_pts_2 = np.hstack((src_pts_2, ones_column))
    dst_pts_2 = np.hstack((dst_pts_2, ones_column))
    ones_column = np.ones((src_pts_3.shape[0], 1))
    src_pts_3 = np.hstack((src_pts_3, ones_column))
    dst_pts_3 = np.hstack((dst_pts_3, ones_column))
    

    match_img_book_1 = cv2.drawMatchesKnn(img_book_1, keypoint_book_1, img_target, keypoint_target,  matches_1, None, 
                               matchColor=(255, 0, 0), singlePointColor=(255, 0, 0), flags=2)
    match_img_book_2 = cv2.drawMatchesKnn(img_book_2, keypoint_book_2, img_target, keypoint_target,  matches_2, None, 
                               matchColor=(255, 0, 0), singlePointColor=(255, 0, 0), flags=2)
    match_img_book_3 = cv2.drawMatchesKnn(img_book_3, keypoint_book_3, img_target, keypoint_target,  matches_3, None, 
                               matchColor=(255, 0, 0), singlePointColor=(255, 0, 0), flags=2)
  
    H_1, mask_1 = get_homography_ransac(src_pts_1, dst_pts_1, num_iterations=1000, threshold=5.0)
    trans_point_1 = np.dot(point_1, H_1.T)
    trans_point_1 = trans_point_1 / trans_point_1[:, [-1]]
    img_book_1_line = draw_line(img_book_1, point_1)
    homo_img_1 = draw_line(img_target.copy(), trans_point_1) 
    inlier_matches_1 = [m for i, m in enumerate(matches_book_1) if mask_1[i] == 1]
    # Draw inlier matches using drawMatchesKnn
    inlier_img_1 = cv2.drawMatchesKnn(img_book_1_line, keypoint_book_1, homo_img_1, keypoint_target,
                                [inlier_matches_1], None, matchColor=(255, 0, 0),
                                singlePointColor=(0, 0, 255), flags=2)
    print(len(inlier_matches_1))
    
    H_2, mask_2 = get_homography_ransac(src_pts_2, dst_pts_2, num_iterations=1000, threshold=5.0)
    trans_point_2 = np.dot(point_2, H_2.T)
    trans_point_2 = trans_point_2 / trans_point_2[:, [-1]]
    img_book_2_line = draw_line(img_book_2, point_2)
    homo_img_2 = draw_line(img_target.copy(), trans_point_2) 
    inlier_matches_2 = [m for i, m in enumerate(matches_book_2) if mask_2[i] == 1]
    # Draw inlier matches using drawMatchesKnn
    inlier_img_2 = cv2.drawMatchesKnn(img_book_2_line, keypoint_book_2, homo_img_2, keypoint_target,
                                [inlier_matches_2], None, matchColor=(255, 0, 0),
                                singlePointColor=(0, 0, 255), flags=2)
    print(len(inlier_matches_2))
    
    H_3, mask_3 = get_homography_ransac(src_pts_3, dst_pts_3, num_iterations=1000, threshold=5.0)
    trans_point_3 = np.dot(point_3, H_3.T)
    trans_point_3 = trans_point_3 / trans_point_3[:, [-1]]
    img_book_3_line = draw_line(img_book_3, point_3)
    homo_img_3 = draw_line(img_target.copy(), trans_point_3) 
    inlier_matches_3 = [m for i, m in enumerate(matches_book_3) if mask_3[i] == 1]
    print(len(inlier_matches_3))
    # Draw inlier matches using drawMatchesKnn
    inlier_img_3 = cv2.drawMatchesKnn(img_book_3_line, keypoint_book_3, homo_img_3, keypoint_target,
                                [inlier_matches_3], None, matchColor=(255, 0, 0),
                                singlePointColor=(0, 0, 255), flags=2)

    in_src_pts_1, in_dst_pts_1 = get_matching_point(np.array(inlier_matches_1), keypoint_book_1, keypoint_target)
    in_src_pts_2, in_dst_pts_2 = get_matching_point(np.array(inlier_matches_2), keypoint_book_2, keypoint_target)
    in_src_pts_3, in_dst_pts_3 = get_matching_point(np.array(inlier_matches_3), keypoint_book_3, keypoint_target)
    ones_column = np.ones((in_src_pts_1.shape[0], 1))
    in_src_pts_1 = np.hstack((in_src_pts_1, ones_column))
    in_dst_pts_1 = np.hstack((in_dst_pts_1, ones_column))
    ones_column = np.ones((in_src_pts_2.shape[0], 1))
    in_src_pts_2 = np.hstack((in_src_pts_2, ones_column))
    in_dst_pts_2 = np.hstack((in_dst_pts_2, ones_column))
    ones_column = np.ones((in_src_pts_3.shape[0], 1))
    in_src_pts_3 = np.hstack((in_src_pts_3, ones_column))
    in_dst_pts_3 = np.hstack((in_dst_pts_3, ones_column))
    
    deviation_img = draw_deviation(img_target.copy(), H_1, in_src_pts_1, in_dst_pts_1, (255,0,0))
    deviation_img = draw_deviation(deviation_img, H_2, in_src_pts_2, in_dst_pts_2, (0,255,0))
    deviation_img = draw_deviation(deviation_img, H_3, in_src_pts_3, in_dst_pts_3, (0,0,255))

    end_time = time.time()
    print(end_time-start_time)

    folder_path = './output/'
    plt.figure(1)
    plt.imshow(cv2.cvtColor(match_img_book_1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    filename = 'matching for book_1.jpg'
    plt.savefig(folder_path + filename)

    plt.figure(2)
    plt.imshow(cv2.cvtColor(match_img_book_2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    filename = 'matching for book_2.jpg'
    plt.savefig(folder_path + filename)
    
    plt.figure(3)
    plt.imshow(cv2.cvtColor(match_img_book_3, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    filename = 'matching for book_3.jpg'
    plt.savefig(folder_path + filename)

    plt.figure(4)
    plt.imshow(cv2.cvtColor(inlier_img_1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    filename = 'optimize matching for book_1.jpg'
    plt.savefig(folder_path + filename)

    plt.figure(5)
    plt.imshow(cv2.cvtColor(inlier_img_2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    filename = 'optimize matching for book_2.jpg'
    plt.savefig(folder_path + filename)

    plt.figure(6)
    plt.imshow(cv2.cvtColor(inlier_img_3, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    filename = 'optimize matching for book_3.jpg'
    plt.savefig(folder_path + filename)

    plt.figure(7)
    plt.imshow(cv2.cvtColor(deviation_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    filename = 'deviation.jpg'
    plt.savefig(folder_path + filename)

    plt.show()

    