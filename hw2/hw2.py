import numpy as np
import cv2

import random
import math
import time

# Function that applies k-means segmentation with k = 10.
# Randomly picks 10 RGB triplets from the existing pixels as initial seeds and runs to convergence.
# After k-means has converged, each cluster is represented with the average RGB value of its members.

def kMeansSeg(grid, k):
    print("Starting to apply k-means segmentation")
    print("Finding 10 random RGB triplets as initial seeds")

    seeds = []
    for i in range(k):
        seeds.append(list(grid[random.sample(range(len(grid)), k)[i]][random.sample(range(len(grid[0])), k)[i]]))

    converged = False
    
    print("Running to Convergence")
    while not converged:
        avg_seeds = [[[0, 0, 0], 0] for i in range(k)]
        clusters = [[] for i in range(k)]

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                smallest_dis = 250000
                index = k + 1
                for z in range(len(clusters)):
                    color_dis = (float(grid[i][j][0]) - float(seeds[z][0])) ** 2 + (float(grid[i][j][1]) - float(seeds[z][1])) ** 2 + (float(grid[i][j][2]) - float(seeds[z][2])) ** 2
                    if color_dis < smallest_dis:
                        index = z
                    smallest_dis = min(color_dis, smallest_dis)
                    
                clusters[index].append([i, j])
                for y in range(3):
                    avg_seeds[index][0][y] += (grid[i][j][y])
                avg_seeds[index][1] += 1

        for i in range(len(clusters)):
            templist = []
            for j in avg_seeds[i][0]:
                templist.append(np.floor(j / avg_seeds[i][1]))
            avg_seeds[i] = templist

        if avg_seeds == seeds:
            converged = True
        seeds = avg_seeds
    
    print("Finished Convergence.")
    final_img = grid.copy()
    
    for i in range(len(clusters)):
        cluster = clusters[i]
        seed = seeds[i]
        for pair in cluster:
            final_img[pair[0]][pair[1]] = seed
   
    return final_img


# Run the SLIC super pixel algorithm on the passed in image
# Segments the image into block_size*block_size segments

def SLIC(grid, size):
    centroids = []
    print("Initializing centroids")
    for i in range(len(grid.astype(int))):
        for j in range(len(grid.astype(int)[i])):
            at_centroid_x = ((i - size/2) % size) == 0
            at_centroid_y = ((j - size/2) % size) == 0
            if at_centroid_x and at_centroid_y and i != 0 and j != 0:
                centroids += [[i, j]]

    print("Computing the magnitude of the gradient and moving the centroids to best position in 3×3 windows")
    for k in range(len(centroids)):
        g_magnitude = 10000000
        cur_coord = [0, 0]
        y = centroids[k][0]
        x = centroids[k][1]
        for i in range(-1, 1):
            for j in range(-1, 1):
                mag1 = float(int(grid[y+i+1][x+j+1][0]) - int(grid[y+i][x+j][0]))**2
                mag2 = float(int(grid[y+i+1][x+j+1][1]) - int(grid[y+i][x+j][1]))**2
                mag3 = float(int(grid[y+i+1][x+j+1][2]) - int(grid[y+i][x+j][2]))**2
                cur_mag = math.sqrt(mag1 + mag2 + mag3)
                if cur_mag < g_magnitude:
                    cur_coord = [y+i, x+j]
                g_magnitude = min(cur_mag, g_magnitude)
        centroids[k] = cur_coord
        
    print("Applying k-means in the 5D space of x, y, R, G, B.")
    centers= [[centroids[i][1], centroids[i][0], grid[centroids[i][0]][centroids[i][1]][2], grid[centroids[i][0]][centroids[i][1]][1], grid[centroids[i][0]][centroids[i][1]][0]] for i in range(len(centroids))]

    finished = False
    while not finished:
        clusters = [[] for i in range(len(centroids))]
        for i in range(len(grid.astype(int))):
            for j in range(len(grid.astype(int)[i])):
                e_distance = 10000000
                index = 0
                for x in range(len(clusters)):
                    center = centers[x]
                    point1 = np.array((center[0], center[1], center[2], center[3], center[4]))
                    point2 = np.array((j, i, grid[i][j][2], grid[i][j][1], grid[i][j][0]))
                    cur_distance = np.sum(np.square(point1 - point2))
                    if cur_distance < e_distance:
                        index = x
                    e_distance = min(cur_distance, e_distance)
                clusters[index].append([j, i, grid[i][j][2], grid[i][j][1], grid[i][j][0]])
        
        print("Finding cluster center")
        for i in range(len(clusters)):
            if len(clusters[i]) == 0:
                continue
            avg_arr = [0, 0, 0]
            for j in range(len(clusters[i])):
                for k in range(len(avg_arr)):
                    avg_arr[k] += int(clusters[i][j][k+2])
            for k in range(len(avg_arr)):
                avg_arr[k] = int(avg_arr[k]/len(clusters[i]))

            if int(avg_arr[0]) != int(centers[i][2]) or int(avg_arr[1]) != int(centers[i][3]) or int(avg_arr[2]) != int(centers[i][4]):
                for k in range(len(avg_arr)):
                    centers[i][k+2] = avg_arr[k]
            else:
                finished = True

    print("Finished SLIC.")
    final_img = grid.copy()
    
    for i in range(len(clusters)):
        cluster = clusters[i]
        center = centers[i]
        for pair in cluster:
            for indx in range(2,5):
                final_img[pair[1]][pair[0]][4-indx] = center[indx]
    return final_img

if __name__ == "__main__":

	# read the white-tower.png file and apply k-means segmentation with k=10
    k = 10
    white_tower_img = cv2.imread("white-tower.png")
    kMeans = kMeansSeg(white_tower_img, k)
    cv2.imwrite("kMeans.png", kMeans)

	# read the wt_slic.png file and apply a variant of the SLIC algorithm
    wt_slic_img = cv2.imread("wt_slic.png")
    applied_slic_img = SLIC(wt_slic_img, 50)
    
    # colors pixels that touch two different clusters black and the remaining pixels by the average RGB value of their cluster
    final_img = applied_slic_img.copy()
    for i in range(len(applied_slic_img)):
        for j in range(len(applied_slic_img[i])):
            if i != 0 and j != 0:
                for row in range(-1, 1):
                    for col in range(-1, 1):
                        diff_clusters = (applied_slic_img[i][j][0] != applied_slic_img[i+row][j+col][0])
                        diff_clusters |= (applied_slic_img[i][j][1] != applied_slic_img[i+row][j+col][1])
                        diff_clusters |= (applied_slic_img[i][j][2] != applied_slic_img[i+row][j+col][2])
                        if diff_clusters:
                            final_img[i][j] = [0, 0, 0]
                            
    cv2.imwrite("wt_slic_applied.png", final_img)