import matplotlib
#matplotlib.use('Agg')
import math
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
import time, sys
import warnings
from joblib import Parallel, delayed
import multiprocessing as m
from tqdm import tqdm
import pandas as pd
import itertools

n_cpu = 12#max(1, m.cpu_count() // 2)
GRID_SIZE = 500
POPULATION = GRID_SIZE ** 2

def is_boundary(mat, pos):
    row, col = pos
    #s = mat[row, col]
    s = 0
    total = 0
    n = mat.shape[0]
    #neighbors = [((i + 1 + n) % n) * n + j, ((i - 1 + n) % n) * n + j, i * n + (j + 1 + n) % n, i * n + (j - 1 + n) % n]
#    neighbors = [((row + 1 + n) % n) * n + col, ((row - 1 + n) % n) * n + col, row * n + (col + 1 + n) % n, row * n + (col - 1 + n) % n, ((row - 1 + n) % n) * n + (col - 1 + n) % n, ((row - 1 + n) % n) * n + (col + 1 + n) % n, ((row + 1 + n) % n) * n + (col - 1 + n) % n, ((row + 1 + n) % n) * n + (col + 1 + n) % n]
#    for neighbor in neighbors:
#        s += mat[neighbor // n, neighbor % n]
    for hor in range(max(col - 1, 0), min(col + 2, GRID_SIZE)):
        for ver in range(max(row - 1, 0), min(row + 2, GRID_SIZE)):
            s += mat[ver, hor]
            total += 1
    return s > 0 and s < total #and mat[row, col] == 0

def filter_inner_boundary(mat_boundary, boundary_list):
    n = mat_boundary.shape[0]
    mat_enclosed = np.zeros((n, n))
    mat_enclosed[0,:] = 1
    mat_enclosed[-1,:] = 1
    mat_enclosed[:,0] = 1
    mat_enclosed[:,-1] = 1
    print("Filtering...")
    for point in tqdm(boundary_list):
        row, col = point
        stack = [row * n + col]
        visited = set([row * n + col])
        while len(stack) > 0:
            p = stack[-1]
            stack = stack[:-1]
            r, c = p // n, p % n
            for hor in range(max(c - 1, 0), min(c + 2, n)):
                for ver in range(max(r - 1, 0), min(r + 2, n)):
                    if r == ver or c == hor:
#                        if hor in [0, n - 1] or ver in [0, n - 1]:
#                            mat_enclosed[row, col] = 1
#                            mat_enclosed[r, c] = 1
                        if mat_enclosed[row, col] == 0:
                            if mat_enclosed[ver, hor] == 1 and mat_boundary[ver, hor] == 0:
                                mat_enclosed[row, col] = 1
                                #mat_enclosed[r, c] = 1
                            elif mat_boundary[ver, hor] == 0:
                                curr = ver * n + hor
                                if curr not in visited:
                                    stack.append(curr)
                                    visited.add(curr)
            if mat_enclosed[row, col] == 1:
                stack = []
    return mat_enclosed

def get_boundary(mat):
    n = mat.shape[0]
    boundary = []
    mat_boundary = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if is_boundary(mat, (i, j)):
                boundary.append((i, j))
                mat_boundary[i, j] = 1
    boundary = set(boundary)
    
    mat_enclosed = filter_inner_boundary(mat_boundary, list(boundary))
    #print(np.sum(mat_enclosed))
    return mat_boundary * mat_enclosed

def sample_graph(n_long=2, seed=0):
    print("Sampling Graph...")
    edges = {}
    np.random.seed(seed)
    long_neighbors = np.random.choice(POPULATION, (POPULATION, n_long))
    for i in tqdm(range(POPULATION)):
        edges[i] = []
        row = i // GRID_SIZE
        col = i % GRID_SIZE
        for hor in range(max(col - 1, 0), min(col + 2, GRID_SIZE)):
            for ver in range(max(row - 1, 0), min(row + 2, GRID_SIZE)):
                edges[i].append(ver * GRID_SIZE + hor)
        edges[i] += list(long_neighbors[i,:])
        edges[i] = list(set(edges[i]))
    return edges

def simulation(edges, p=0.6, T=10, m=7):
    n = GRID_SIZE
    center = n // 2
    mat = np.zeros((n, n))
    mat[center, center] = 1
    q = Queue()
    boundary_num = []
    infected_num = []
    print("Simulating...")
    for _ in tqdm(range(T)):
        for i in range(n):
            for j in range(n):
                if mat[i, j] > 0 and mat[i, j] <= m:
                    q.put(i * n + j)
        while not q.empty():
            point = q.get()
            row = point // n
            col = point % n
            if mat[row, col] > 0 and mat[row, col] <= m:
                #neighbors = [((row + 1 + n) % n) * n + col, ((row - 1 + n) % n) * n + col, row * n + (col + 1 + n) % n, row * n + (col - 1 + n) % n]
                neighbors = edges[point]
                for neighbor in neighbors:
                    i = neighbor // n
                    j = neighbor % n
                    dist = abs(i - row) + abs(j - col)
                    if mat[i, j] == 0 and np.random.uniform() < p / dist ** 1:
                        mat[i, j] = 1
                mat[row, col] += 1
                
#        mat_curr = (mat > 0) + 0
#        mat_boundary_curr = get_boundary(mat_curr)
#        boundary_num.append(np.sum(mat_boundary_curr))
#        infected_num.append(np.sum(mat_curr))
        
    mat = (mat > 0) + 0
    mat_boundary = get_boundary(mat)
    
#    plt.plot(infected_num, boundary_num)
#    plt.xlabel("Infected People")
#    plt.ylabel("Boundary People")
#    plt.savefig("../Plots/Ratio.png")
    return mat, mat_boundary

def plot_matrices(orig_mat, boundary_mat):
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    ax[0].imshow(orig_mat, cmap="hot", interpolation="nearest")
    ax[1].imshow(boundary_mat, cmap="hot", interpolation="nearest")
    ax[0].set_title("Infection Status\n# Infected People = " + str(np.sum(orig_mat)))
    ax[1].set_title("True Boundary\nTrue # Boundary People = " + str(np.sum(boundary_mat)))
    plt.savefig("../Plots/Boundary.tiff", dpi=300)
        
edges = sample_graph(n_long=1, seed=1)
mat, mat_boundary = simulation(edges, p=0.075, T=120, m=7)
plot_matrices(mat, mat_boundary)
