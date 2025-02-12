import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


def sift_extractor(img):
    # initializes the SIFT
    sift = cv2.SIFT_create()
    # compute keypoints and descriptors for input image
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def feature_matching(keypoint_target, desc_source, desc_target):

    print('matching...')
    matches_ob1 = []
    matches_ob2 = []
    matches_ob3 = []
    ratio_threshold = 0.7
    
    # loop through all the descriptors of source image to find the best match in target image
    for i, desc1 in enumerate(desc_source):
        best_match = None

        # find distances between desc1 and all the descriptors of target image
        distances = np.linalg.norm(desc1 - desc_target, axis=1)

        # ratio testing
        indices = list(range(len(distances)))
        # sort the distances in order to find the best(sorted_indices[0])and the second best(sorted_indices[1])
        sorted_indices = sorted(indices, key=lambda i: distances[i])
        best_distance = distances[sorted_indices[0]]
        second_best_distance = distances[sorted_indices[1]]
        # if the best distance is smaller than 0.7*second best distance, then we consider it as a good match and store it
        if best_distance < ratio_threshold * second_best_distance:
           # i: index if source keypoint, sorted_indices[0]: index of target keypoint, best_distance: distance between two keypoint 
           best_match = cv2.DMatch(i, sorted_indices[0], best_distance) 

        # if the match fit the ratio test, we then check which object the match belongs to and store it to the corresponding list
        if best_match is not None:
            y_pos = keypoint_target[sorted_indices[0]].pt[1]
            if y_pos >= 100 and y_pos <= 600:   # 100-600 is the bounday of the first object
                matches_ob1.append(best_match)   
            elif y_pos >= 1000 and y_pos <= 1500:   # 1000-1500 is the boundary of the second object
                matches_ob2.append(best_match)
            elif y_pos >= 1600: # 2000-the end of the picture is the boundary of the third object
                matches_ob3.append(best_match)
    
    print('sorting...')
    # sort the matches for each object according to the distance
    matches_ob1 = sorted(matches_ob1, key=lambda x: x.distance) 
    matches_ob2 = sorted(matches_ob2, key=lambda x: x.distance) 
    matches_ob3 = sorted(matches_ob3, key=lambda x: x.distance)

    # total matches only get the top 20 matches for each object
    total_matches = matches_ob1[0:20] + matches_ob2[0:20] + matches_ob3[0:20]
    return total_matches

if __name__ == '__main__':
    
    start_time = time.time()
    img_source = cv2.imread('hw1-3-1.jpg', cv2.IMREAD_GRAYSCALE)
    img_target = cv2.imread('hw1-3-2.jpg', cv2.IMREAD_GRAYSCALE)
    
    keypoint_source, desc_source = sift_extractor(img_source)
    keypoint_target, desc_target = sift_extractor(img_target)
    
    best_matches  = feature_matching(keypoint_target, desc_source, desc_target)
    best_matches = [[match] for match in best_matches]
    img_match = cv2.drawMatchesKnn(img_source, keypoint_source, img_target, keypoint_target,  best_matches, None, 
                               matchColor=(255, 0, 0), singlePointColor=(255, 0, 0), flags=2)
    
    #########################################################################################################
    # Addition setting: scaled image
    img_resize = cv2.resize(img_source, None , interpolation=cv2.INTER_LINEAR,  fx=2, fy=2) 
    keypoint_resize, desc_resize = sift_extractor(img_resize)
    best_matches_resize  = feature_matching(keypoint_target, desc_resize, desc_target)
    best_matches_resize = [[match] for match in best_matches_resize]
    img_match_resize = cv2.drawMatchesKnn(img_resize, keypoint_resize, img_target, keypoint_target,  best_matches_resize, None, 
                               matchColor=(255, 0, 0), singlePointColor=(255, 0, 0), flags=2)
    
    end_time = time.time()
    print('time:', end_time-start_time)
    
    plt.figure(1)
    plt.imshow(img_match)
    plt.title('Original')
    plt.axis('off')

    plt.figure(2)
    plt.imshow(img_match_resize)
    plt.title('Scaled')
    plt.axis('off')

    plt.imsave('./result_image/hw1-3(c.).jpg', img_match)
    plt.imsave('./result_image/hw1-3(d.).jpg', img_match_resize)
    plt.show()

    