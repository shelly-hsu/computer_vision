import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import time
from sklearn.cluster import MeanShift
import torch
from sklearn.cluster import DBSCAN
from PIL import Image

def kmeans_plus_init(data, k):
    num_points, _ = data.shape
    centers =  []

    # Randomly choose the first center from the data points
    rand_index = np.random.choice(num_points)
    centers.append(data[rand_index])
    
    # Calculate distances for subsequent centroids
    for _ in range(1, k):
        distances = np.linalg.norm(data[:, None] - np.array(centers), axis=2) ** 2
        min_distances = np.min(distances, axis=1)
        probabilities = min_distances / np.sum(min_distances)
        cumulative_probabilities = np.cumsum(probabilities)
        rand = np.random.rand()
        
        # Choose the next centroid with a probability proportional to its squared distance
        for i, prob in enumerate(cumulative_probabilities):
            if prob > rand:
                centers.append(data[i])
                break
    return np.array(centers)

def kmeans(img, k, clustering_type, tolerance_threshold=1e-4):
    print('kmeans....')
    reshaped_img = img.reshape((-1, 3)).astype(np.float32)  # Reshape the image
    num_points, _ = reshaped_img.shape
    best_center = None
    best_obj_func = np.inf  # Initialize with a large value
    for iteration in range(50):  # Perform multiple initializations
        print('iteration', iteration)
        if clustering_type == 'plus':
            center = kmeans_plus_init(reshaped_img, k)
        else:
            center = reshaped_img[np.random.choice(num_points, k, replace=False)]
        while True:  # Perform iterations for convergence
            distances = np.linalg.norm(reshaped_img[:, None] - center, axis=2)
            cluster_assignment = np.argmin(distances, axis=1)
            prev_center = center.copy()  # Make a copy of centroids before updating
            for i in range(k):  # Assign points to corresponding clusters
                points_for_center = reshaped_img[cluster_assignment == i]
                if len(points_for_center) > 0:
                    center[i] = np.mean(points_for_center, axis=0)
            # Check convergence by measuring centroid changes
            centroid_change = np.linalg.norm(center - prev_center)
            if centroid_change < tolerance_threshold:  # Check if centroids have converged
                # print('break')
                break  # Stop iterations if centroids have converged
        # Calculate objective function (Sum of Squared Errors)
        distances = np.linalg.norm(reshaped_img[:, None] - center, axis=2)
        obj_func = np.sum(np.min(distances, axis=1))  # SSE
        # Update best_center and best_obj_func if the current result is better
        if obj_func < best_obj_func:
            best_obj_func = obj_func
            best_center = center.copy()
    print('finish')
    # Assign each pixel to its closest centroid using the best_center found
    distances = np.linalg.norm(reshaped_img[:, None] - best_center, axis=2)
    cluster_assignment = np.argmin(distances, axis=1)
    segmented_img = best_center[cluster_assignment]
    segmented_img = segmented_img.reshape(img.shape).astype(np.uint8)
    return segmented_img

def plot_pixel_distribution(data, title):
    print('plot')
    global count
    folder_path = './output/'
    fig = plt.figure(count)
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    # Extract RGB values
    red_values = data[:, 0]
    green_values = data[:, 1]
    blue_values = data[:, 2]
    # Create a 3D scatter plot for R * G * B feature space
    ax.scatter(red_values, green_values, blue_values, c=data / 255.0, s=5) 
    
    if count == 1:
        title_fig = 'img1_' + title 
        filepath = './output/' + 'img1_' + title + '.jpg'
    else:
        title_fig = 'img2_' + title
        filepath= './output/' + 'img2_' + title + '.jpg'
    plt.title(title_fig)
    plt.savefig(filepath)
    plt.close()

def plot_pixel_distribution_after(img, centers, labels, title):
    
    global count
    folder_path = './output/'
    data = img.reshape(-1,3)
    colors = centers[labels]
    fig = plt.figure(count)
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    # Extract RGB values
    red_values = data[:, 0]
    green_values = data[:, 1]
    blue_values = data[:, 2]
    # Create a 3D scatter plot for R * G * B feature space
    ax.scatter(red_values, green_values, blue_values, c=colors/255, s=5) 
    if count == 1:
        title_fig = 'img1_' + title 
        filepath = './output/' + 'img1_' + title + '.jpg'
    else:
        title_fig = 'img2_' + title
        filepath= './output/' + 'img2_' + title + '.jpg'
    plt.title(title_fig)
    plt.savefig(filepath)
    plt.close()


def mean_shift_gpu(data, bandwidth):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    centroids = torch.clone(data_tensor)
    iteration = 0

    while True:
        print('iteration', iteration)
        
        new_centroids = []
        iteration += 1

        for centroid in centroids:
            distances = torch.norm(data_tensor - centroid, dim=1)
            in_bandwidth = data_tensor[distances <= bandwidth]
            
            if len(in_bandwidth) > 0:
                new_centroid = torch.mean(in_bandwidth, dim=0)
                new_centroids.append(new_centroid)

        new_centroid_np = [centroid.cpu().detach().numpy() for centroid in new_centroids]
        centroids_np = centroids.cpu().detach().numpy()  # Convert PyTorch tensor to NumPy array

        # Calculate element-wise distances between arrays
        element_wise_distances_np = np.abs(new_centroid_np - centroids_np)

        # Check if all element-wise distances are less than 1e-5
        condition_met = np.all(element_wise_distances_np < 1e-5)

        if (condition_met == True) or iteration >= 1000:
            break
        centroids = torch.unique(torch.stack(new_centroids), dim=0)
        print('len',len(centroids))
    
    ###########################################################################################
    new_centroids = [centroid.cpu().numpy() for centroid in new_centroids]
    stacked_centroids_np = np.stack(new_centroids)

    # Calculate pairwise distances between centroids using broadcasting
    distances = np.sqrt(np.sum((stacked_centroids_np[:, None] - stacked_centroids_np) ** 2, axis=-1))

    # Set the diagonal and upper triangle values to infinity to exclude self-comparisons and duplicates
    np.fill_diagonal(distances, np.inf)
    distances[np.triu_indices(len(new_centroids))] = np.inf

    # Set a threshold for similarity
    threshold = bandwidth  # Set your threshold here

    # Get indices of centroids that are sufficiently unique based on the threshold
    unique_indices = np.where(np.min(distances, axis=0) >= threshold)[0]

    # Gather unique centroids based on unique indices
    unique_centroids = np.array([torch.tensor(new_centroids[i]) for i in unique_indices])
    print('cluster number:',len(unique_centroids))
    print('center:', unique_centroids)
    return unique_centroids

def predict(data, centroids):
    data_points_np = np.array(data)
    centroids_np = np.array(centroids)
    # Calculate pairwise distances between data points and centroids
    distances = np.linalg.norm(data_points_np[:, None] - centroids_np, axis=-1)
    # Find the index of the closest centroid for each data point
    labels = np.argmin(distances, axis=1)
    return labels

if __name__ == '__main__':

    start_time = time.time()
    global count
    count = 1
    # Read the image in RGB format
    img1 = cv2.imread('2-image.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread('2-masterpiece.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_list = [img1, img2]
    k_values = [4, 6, 8]
    bandwidth_set =[90, 60, 30]

    # kmean and kmean++
    for i, img in enumerate(img_list):
        for k in k_values:
            kmeans_img = kmeans(img, k, 'normal')
            kmeans_plus_img = kmeans(img, k, 'plus')

            filepath_1 = './output/' + 'img' + str(i+1) + '_kmeans_k=' + str(k) + '.jpg'
            plt.imsave(filepath_1, kmeans_img)
        
            filepath_2 = './output/' + 'img' + str(i+1) + '_kmeans_plus_k='  + str(k) + '.jpg'
            plt.imsave(filepath_2, kmeans_plus_img)

    # mean shift
    for i, img in enumerate(img_list):
        data = img.reshape(-1,3).astype(np.float64)
        plot_pixel_distribution(data,  '_Pixel Distribution Before Mean Shift')

        for bandwidth in bandwidth_set:

            print('bandwidth:', bandwidth)
            center = mean_shift_gpu(data, bandwidth)
            label = predict(data, center)
            segmented_data = center[label]
            segmented_img = segmented_data.reshape(img.shape).astype(np.uint8)
            if bandwidth == 30:
                plot_pixel_distribution(segmented_data, str(bandwidth) + '_clustering resulting')
                plot_pixel_distribution_after(img, center, label, str(bandwidth) + '_Pixel Distribution After Mean Shift')
            
            filepath =  './output/' + 'img' + str(count) + '_' + str(bandwidth) +'_mean_shifted.jpg'
            plt.imsave(filepath, segmented_img)
            
        print('channels_5')
        image_5channel = np.zeros((img.shape[0], img.shape[1], 5), dtype=np.uint8)
        image_5channel[:, :, :3] = img
        x_coords, y_coords = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        # Assign the spatial information to the new channels
        image_5channel[:, :, 3] = x_coords.astype(np.uint8)
        image_5channel[:, :, 4] = y_coords.astype(np.uint8)  
        print('spatial_mean_shifting...')
        spatial_data = image_5channel.reshape(-1,5).astype(np.float64)
        spatial_center = mean_shift_gpu(spatial_data, 90)
        spatial_label = predict(spatial_data, spatial_center)
        segmented_data = spatial_center[spatial_label]
        segmented_img = segmented_data[:,:3].reshape(img.shape).astype(np.uint8)
        filepath =  './output/' + 'img' + str(count) + '_' + 'spatial_mean_shift.jpg'
        plt.imsave(filepath, segmented_img)
        count+=1
            
    end_time = time.time()
    print(end_time-start_time)
    
