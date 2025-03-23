import cv2
import numpy as np
import copy
import os
# mouse callback function
def mouse_callback(event, x, y, flags, param):
    
    global corner_list
    if event == cv2.EVENT_LBUTTONDOWN:  
        if(len(corner_list)<4):
            corner_list.append((x, y))
        
def Find_Homography(world,camera):
    
    # Create the A matrix
    A = []
    for i in range(4):
        x, y = world[i]
        u, v = camera[i]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])

    A = np.array(A)

    # Perform singular value decomposition (SVD) on A
    U, D, Vt = np.linalg.svd(A)

    # The homography matrix H is the last column of Vt
    H = Vt[-1, :].reshape(3, 3)

    # Normalize H by dividing by H[2,2]
    H = H / H[2, 2]

    return H

if __name__=="__main__":
    
    img_src = cv2.imread("assets/post.png") 
    src_H, src_W, _ = img_src.shape
    file_path = "./output"
    img_tar = cv2.imread("assets/display.jpg") 
    
    top_left = (0, 0)
    top_right = (src_W, 0)
    bottom_left = (0, src_H)
    bottom_right = (src_W, src_H)
    src_points = np.array([top_left, top_right, bottom_left, bottom_right], np.float64)
    
    cv2.namedWindow("Interative window")

    cv2.setMouseCallback("Interative window", mouse_callback)
    cv2.setMouseCallback("Interative window", mouse_callback)
    
    corner_list = []
    while True:
        fig=img_tar.copy()
        key = cv2.waitKey(1) & 0xFF
        
        if(len(corner_list)==4):

            des_points = np.array(sorted(corner_list, key=lambda x: x[1]), dtype=np.float64)
            if(des_points[0][0] > des_points[1][0]):
                des_points[0], des_points[1] = des_points[1].copy(), des_points[0].copy()
            if(des_points[2][0] < des_points[3][0]):
                des_points[2], des_points[3] = des_points[3].copy(), des_points[2].copy()

            # implement the inverse homography mapping and bi-linear interpolation 
            # testing_H = Find_Homography(src_points, des_points)
            H, _ = cv2.findHomography(src_points, des_points, cv2.RANSAC, 5.0)

            warped_image = cv2.warpPerspective(img_src, H, (img_tar.shape[1], img_tar.shape[0])) 

            # # fig = map_image_to_coordinates(img_src, img_src, H, sorted_list)
        


        # quit 
        if key == ord("q"):
            break
        # reset the corner_list
        if key == ord("r"):
            corner_list=[]
        # show the corner list
        if key == ord("p"):
            print(des_points)
            # print(testing_H)
            print(H)
            cv2.imshow("Mapped Image", warped_image)

        cv2.imshow("Interative window", fig)
    cv2.imwrite(os.path.join(file_path,"homography.png"),fig)
    cv2.destroyAllWindows()