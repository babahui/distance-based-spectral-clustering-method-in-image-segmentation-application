"""
=========================
our image segmentation method
=========================


"""

print(__doc__)

import numpy as np
import scipy
from skimage.segmentation import slic, felzenszwalb
from skimage.future import graph
import networkx as nx
from scipy import sparse
import matplotlib
import matplotlib.pyplot as plt
from admm_algorithms import admm, relation_density_admm, var_admm

from copy import deepcopy
import math

from mpl_toolkits.mplot3d import Axes3D
from skimage.future import graph as gg
from skimage import data, segmentation, color

from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.io import imsave

import pylab as pl
from os.path import join
import os

from numpy import linalg as LA
from sklearn.preprocessing import normalize
from imageio import imread

from scipy.spatial import distance
from skimage.feature.texture import local_binary_pattern
import time
import warnings


def EM_method(img_path, sp_met='felzenszwalb', graph_met='syn_met', admm_met='admm', num_cuts=3, dist_hist=False, lambda_coff=False, n_iter=1000, K=None, EM_iter=3):
   
    m_img = imread(img_path)
    
    # superpixels method
    if sp_met == 'felzenszwalb':
        segments = felzenszwalb(m_img, scale=10, sigma=0.5, min_size=100)
    elif sp_met == 'slic':
        segments = slic(m_img, compactness=30, n_segments=400)
    else:
        warnings.warn("Warning Message: no superpixels method parameter")
        
    ave_position = init_sp(m_img, segments)
        
    ev, distance_matrix = syn_graph_met(m_img, segments, lambda_coff=lambda_coff, dist_hist=dist_hist)   
    

    iteration = 0
    while iteration < EM_iter:
        # iterate step1 and step2
        
        # global segmentation, return labels
        
        vals, vectors = np.linalg.eigh(distance_matrix)
        vals, vectors = np.real(vals), np.real(vectors)
        index1 = np.argsort(vals)[0]
        ev = vectors[:, index1]

        sp_label = admm(n_vector=ev, n_iter=n_iter, num_cuts=num_cuts)
        p_label, labels = pixels_label(m_img, segments, sp_label)
    
    
        # local constraint, return distance matrix
#         if not K:
#              K = 50
        distance_matrix = local_constraint(distance_matrix, ave_position, K, sp_label)
        
        iteration += 1

        # plot

            
    return p_label

        
def local_constraint(distance_matrix, ave_position, K, sp_label):
    # if sps in differnet labels and distance < a, change labels or distance
    length = len(ave_position)
    for i in range(length):
        for j in range(length):
            if LA.norm(np.asarray(ave_position[i]) - np.asarray(ave_position[j])) < K and sp_label[i] != sp_label[j]:
                diff = 0
                distance_matrix[i, j] = diff
            
    return distance_matrix


def init_sp(image, segments):

    # init graph by first method, by color distance metric between superpixels.
    row = image.shape[0]
    col = image.shape[1]

    segmentsLabel = []
    for i in range(row):
        for j in range(col):
            l = segments[i, j]
            if l not in segmentsLabel:
                segmentsLabel.append(l)
    position = []
    ave_position = []
    flatten_position = []
    
    for i in segmentsLabel:
        pixel_position = []
        flatten_pos = []
        for m in range(row):
            for n in range(col):
                if segments[m, n] == i:
                    pixel_position.append([m, n])
                    flatten_pos.append(m * col + n)
                    
        position.append(pixel_position)
        flatten_position.append(flatten_pos)
        
        pixel_position = np.asarray(pixel_position)
        ave_position.append((sum(pixel_position) / len(pixel_position)).tolist())
        
    # generate average color value and red, green, blue color values
    average = []
    red_average = []
    green_average = []
    blue_average = []
    for i in range(len(position)):
        val = 0
        red_val = 0
        green_val = 0
        blue_val = 0
        for j in position[i]:
            [m, n] = j
            val += 0.299*image[m, n, 0] + 0.587*image[m, n, 1] + 0.114*image[m, n, 2]
            red_val += image[m, n, 0]
            green_val += image[m, n, 1]
            blue_val += image[m, n, 2]
            # val += image[m, n]
        average.append(val/len(position[i]))
        red_average.append(red_val/len(position[i]))
        green_average.append(green_val/len(position[i]))
        blue_average.append(blue_val/len(position[i]))
    
    return ave_position

        
def image_seg(img_path, sp_met='felzenszwalb', graph_met='syn_met', admm_met='admm', num_cuts=3, dist_hist=False, lambda_coff=False, n_iter=1000):
    m_img = imread(img_path)
    
    # superpixels method
    if sp_met == 'felzenszwalb':
        segments = felzenszwalb(m_img, scale=10, sigma=0.5, min_size=100)
    elif sp_met == 'slic':
        segments = slic(m_img, compactness=30, n_segments=400)
    else:
        warnings.warn("Warning Message: no superpixels method parameter")
   

    # generate graph matrix
    if graph_met == 'lib_met':
        g = graph.rag_mean_color(m_img, segments)
        w = nx.to_scipy_sparse_matrix(g, format='csc')
        entries = w.sum(axis=0)
        d = sparse.dia_matrix((entries, 0), shape=w.shape).tocsc()
        m = w.shape[0]
        d2 = d.copy()
        d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)
        matrix = d2 * (d - w) * d2

        # matrix eigen-decomposition, scipy.sparse.linalg
        vals, vectors = scipy.sparse.linalg.eigsh(matrix, which='SM', k=min(100, m - 2))
        vals, vectors = np.real(vals), np.real(vectors)
        index1, index2, index3 = np.argsort(vals)[0], np.argsort(vals)[1], np.argsort(vals)[2]
        ev1, ev, ev3 = vectors[:, index1], vectors[:, index2], vectors[:, index3]
        
    elif graph_met == 'syn_met':
        ev = syn_graph_met(m_img, segments, lambda_coff=lambda_coff, dist_hist=dist_hist)     
        
    else:
        warnings.warn('Warning Message: graph_met argument missing')
        
        
            
    if admm_met == 'admm':
        sp_label = admm(n_vector=ev, n_iter=n_iter, num_cuts=num_cuts)

    elif admm_met == 'density_admm':
        sp_label = relation_density_admm(n_vector=ev, num_cuts=num_cuts)

    else:
        warnings.warn('Warning Message: admm_met argument missing')

   
    p_label, labels = pixels_label(m_img, segments, sp_label)
    return p_label


def before_method(img_path, sp_met='felzenszwalb', graph_met='syn_met', admm_met='admm', num_cuts=3, dist_hist=False, lambda_coff=False, n_iter=1000):
    m_img = imread(img_path)
    
    # superpixels method
    if sp_met == 'felzenszwalb':
        segments = felzenszwalb(m_img, scale=10, sigma=0.5, min_size=100)
    elif sp_met == 'slic':
        segments = slic(m_img, compactness=30, n_segments=400)
    else:
        warnings.warn("Warning Message: no superpixels method parameter")
   

    # generate graph matrix
    if graph_met == 'lib_met':
        g = graph.rag_mean_color(m_img, segments)
        w = nx.to_scipy_sparse_matrix(g, format='csc')
        entries = w.sum(axis=0)
        d = sparse.dia_matrix((entries, 0), shape=w.shape).tocsc()
        m = w.shape[0]
        d2 = d.copy()
        d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)
        matrix = d2 * (d - w) * d2

        # matrix eigen-decomposition, scipy.sparse.linalg
        vals, vectors = scipy.sparse.linalg.eigsh(matrix, which='SM', k=min(100, m - 2))
        vals, vectors = np.real(vals), np.real(vectors)
        index1, index2, index3 = np.argsort(vals)[0], np.argsort(vals)[1], np.argsort(vals)[2]
        ev1, ev, ev3 = vectors[:, index1], vectors[:, index2], vectors[:, index3]
        
    elif graph_met == 'syn_met':
        ev = syn_graph_met(m_img, segments, lambda_coff=lambda_coff, dist_hist=dist_hist)     
        
    else:
        warnings.warn('Warning Message: graph_met argument missing')
        
        
            
    if admm_met == 'admm':
        sp_label = admm(n_vector=ev, n_iter=n_iter, num_cuts=num_cuts)

    elif admm_met == 'density_admm':
        sp_label = relation_density_admm(n_vector=ev, num_cuts=num_cuts)

    else:
        warnings.warn('Warning Message: admm_met argument missing')

   
    p_label, labels = pixels_label(m_img, segments, sp_label)
    return p_label



def pixels_label(m_img, segments, sp_label):
    # get superpixels position
    row, col = m_img.shape[0], m_img.shape[1]
    seg_label = []
    for i in range(row):
        for j in range(col):
            l = segments[i, j]
            if l not in seg_label:
                seg_label.append(l)
    sp_pos = []
    for i in seg_label:
        i_pos = []
        for m in range(row):
            for n in range(col):
                if segments[m, n] == i:
                    i_pos.append([m, n])
        sp_pos.append(i_pos)

    labels = []
    for i in range(len(sp_label)):
        if sp_label[i] not in labels:
            labels.append(sp_label[i])

    p_label = np.ones((row, col))
    for i in range(len(labels)):
        label_i_index = [j for j in range(len(sp_label)) if sp_label[j] == labels[i]]
        label_i_p = []
        for k in label_i_index:
            label_i_p.extend(sp_pos[k])
        color = int(i)
        for p in label_i_p:
            [cor_i, cor_j] = p
            p_label[cor_i, cor_j] = color

    p_label = np.asarray(p_label, dtype=int)

    return p_label, labels

    
def syn_graph_met(m_img, segments, lambda_coff, dist_hist=False):
    image = m_img

    # init graph by first method, by color distance metric between superpixels.
    row = image.shape[0]
    col = image.shape[1]
#     print(row, col)
    segmentsLabel = []
    for i in range(row):
        for j in range(col):
            l = segments[i, j]
            if l not in segmentsLabel:
                segmentsLabel.append(l)
    
    position = []
    ave_position = []
    flatten_position = []
    
    for i in segmentsLabel:
        pixel_position = []
        flatten_pos = []
        for m in range(row):
            for n in range(col):
                if segments[m, n] == i:
                    pixel_position.append([m, n])
                    flatten_pos.append(m * col + n)
                    
        position.append(pixel_position)
        flatten_position.append(flatten_pos)
        
        pixel_position = np.asarray(pixel_position)
        ave_position.append((sum(pixel_position) / len(pixel_position)).tolist())
        
    # generate average color value and red, green, blue color values
    average = []
    red_average = []
    green_average = []
    blue_average = []
    for i in range(len(position)):
        val = 0
        red_val = 0
        green_val = 0
        blue_val = 0
        for j in position[i]:
            [m, n] = j
            val += 0.299*image[m, n, 0] + 0.587*image[m, n, 1] + 0.114*image[m, n, 2]
            red_val += image[m, n, 0]
            green_val += image[m, n, 1]
            blue_val += image[m, n, 2]
            # val += image[m, n]
        average.append(val/len(position[i]))
        red_average.append(red_val/len(position[i]))
        green_average.append(green_val/len(position[i]))
        blue_average.append(blue_val/len(position[i]))

    # distance metric: by average value
    # average = []
    # for i in range(len(position)):
    #     val = 0
    #     for j in position[i]:
    #         [m, n] = j
    #         val += 0.299*image[m, n, 0] + 0.587*image[m, n, 1] + 0.114*image[m, n, 2]
    #         # val += image[m, n]
    #     average.append(val/len(position[i]))

    # length = len(position)
    # graph = np.zeros((length, length))
    # for i in range(length):
    #     for j in range(length):
    #         graph[i, j] = abs(average[i] - average[j]) ** 2
    
    graph_time = time.time()
    
    # fully connected
    sigma = 255.0
    length = len(position)
    graph = np.zeros((length, length))
    
    # settings for LBP
    radius = 2
    n_points = 8 * radius
    METHOD = 'uniform'
    
    img, lbp = [], []
    for i in range(3):
        c_img = image[:,:,i]
        c_lbp = local_binary_pattern(c_img, n_points, radius, METHOD)
        img.append(c_img)
        lbp.append(c_lbp)
    
    for i in range(length):
        for j in range(length):     
            if not dist_hist:
                diff = abs(red_average[i]-red_average[j]) + abs(green_average[i]-green_average[j]) + abs(blue_average[i]-blue_average[j])
#                 if lambda_coff:
#                     dist = LA.norm(np.asarray(ave_position[i]) - np.asarray(ave_position[j]))
#                     diff = diff + lambda_coff * dist 
            else:
                # reads an input image, color mode
                hist1 = hist(flatten_position[i], img, lbp)
                hist2 = hist(flatten_position[j], img, lbp)
                
                diff = abs(distance.cityblock(hist1, hist2))
                
            graph[i, j] = diff
            # graph[i, j] = math.e ** (-(diff ** 2) / sigma)

#     print('graph construction time: ', time.time() - graph_time)    
    
    # matrix eigen-decomposition, scipy.sparse.linalg
    vals, vectors = np.linalg.eigh(graph)
    vals, vectors = np.real(vals), np.real(vectors)
    index1, index2, index3 = np.argsort(vals)[0], np.argsort(vals)[1], np.argsort(vals)[2]
    # index1, index2, index3 = np.argsort(vals)[::-1][0], np.argsort(vals)[::-1][1], np.argsort(vals)[::-1][2]
    ev1, ev2, ev3 = vectors[:, index1], vectors[:, index2], vectors[:, index3]
    
    return vectors[:, index1], graph



def hist(position, img, lbp):
    # color hist: for each channel * 16 bins
    # find frequency of pixels in range 0-255, calculate histogram of blue, green or red channel respectively.
    color_hist = []
    for i in range(3):
        c_img = img[i]
        sp_arr = np.take(c_img, position)
                
        histr, bins = np.histogram(sp_arr, bins=np.linspace(0, 256, num=21))
#         histr = np.reshape(histr, (16, 16)) # 16 bins
#         histr = np.sum(histr, axis=1)  
        color_hist.append(histr)
        
    color_hist = normalize(color_hist).flatten()
    color_hist = np.asarray(color_hist)
    
    # texture hist, for each channel, orientation * 10 bins
    texture_hist = []
    for i in range(3):
        c_lbp = lbp[i]
        sp_lbp = np.take(c_lbp, position)
            
        n_bins = int(sp_lbp.max() + 1)
        histr, _ = np.histogram(sp_lbp, density=True, bins=np.linspace(0, n_bins, num=11))
        texture_hist.append(histr)
    
    texture_hist = normalize(texture_hist).flatten()
    texture_hist = np.asarray(texture_hist)   
        
    hist = np.append(color_hist, texture_hist)
#     print('-----hist--------', hist.shape)
    return np.append(color_hist, texture_hist)
    


