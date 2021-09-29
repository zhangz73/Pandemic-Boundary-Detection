import matplotlib
matplotlib.use('Agg')
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
import numpy as np
from queue import Queue
import time, sys
import warnings
from joblib import Parallel, delayed
import multiprocessing as m
from tqdm import tqdm
import pandas as pd
import itertools, functools

n_cpu = 30#max(1, m.cpu_count() // 2)
GRID_SIZE = 500

POPULATION = 250000#GRID_SIZE ** 2
np.seterr(divide='ignore', invalid='ignore')

def sample_graph(POPULATION = POPULATION, GRID_SIZE = GRID_SIZE):
    horizontal = np.random.rand(POPULATION) * GRID_SIZE
    vertical = np.random.rand(POPULATION) * GRID_SIZE
    horizontal[0] = GRID_SIZE / 2
    vertical[0] = GRID_SIZE / 2
    Locations = {}
    Grid_locs = {}
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            Grid_locs[(i, j)] = []
    for i in range(POPULATION):
        Locations[i] = (horizontal[i], vertical[i])
        Grid_locs[(int(vertical[i]), int(horizontal[i]))].append(i)
        
    Edges = {}
    for i in range(POPULATION):
        Edges[i] = []
        for hor in range(max(int(horizontal[i]) - 1, 0), min(int(horizontal[i]) + 2, GRID_SIZE)):
            for ver in range(max(int(vertical[i]) - 1, 0), min(int(vertical[i]) + 2, GRID_SIZE)):
                if True: #hor == int(horizontal[i]) or ver == int(vertical[i]):
                    Edges[i] += Grid_locs[(ver, hor)]
    return Locations, Edges, Grid_locs

def get_boundary_single(grid_infected, i, j, mat_pred = None, bar = 0):
    GRID_SIZE = grid_infected.shape[0]
    s = 0
    deg = 0
    for k in range(max(i - 1, 0), min(i + 2, GRID_SIZE)):
        for l in range(max(j - 1, 0), min(j + 2, GRID_SIZE)):
            deg += 1
            if mat_pred is not None and mat_pred[k, l] == 0:
                if grid_infected[k, l] > bar:
                    mat_pred[k, l] = 1
                else:
                    mat_pred[k, l] = -1
            if grid_infected[k, l] > bar:
                s += 1
    return (s > 0 and s < deg) + 0

def get_boundary(grid_infected):
    GRID_SIZE = grid_infected.shape[0]
    grid_boundary = np.zeros((GRID_SIZE, GRID_SIZE))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            grid_boundary[i, j] = get_boundary_single(grid_infected, i, j)
    return grid_boundary
    
def get_boundary_external(mat, mat_boundary):
    visited = set([])
    reachable = set([])
    n = mat_boundary.shape[0]
    mat_boundary_external = np.zeros((n, n))
#    for i in range(n):
#        reachable.add(i)
#        reachable.add((n - 1) * n + i)
#        reachable.add(i * n)
#        reachable.add(i * n + n - 1)
    for i in range(n):
        for j in range(n):
            idx = j
            if mat_boundary[i, j] == 0:
                reachable.add(i * n + j)
            else:
                break
        for j in range(n - 1, idx, -1):
            if mat_boundary[i, j] == 0:
                reachable.add(i * n + j)
            else:
                break
    for i in range(n):
        for j in range(n):
            curr = i * n + j
            if mat_boundary[i, j] == 1:
                stack = [curr]
                visited_curr = set([])
                while len(stack) > 0:
                    point = stack.pop()
                    row = point // n
                    col = point % n
                    visited_curr.add(point)
                    if point in reachable:
                        mat_boundary_external[i, j] = 1
                        #reachable.add(curr)
#                         for p in visited_curr:
#                            if mat_boundary[p // n, p % n] == 0:
#                                reachable.add(p)
                        while len(stack) > 0:
                            p = stack.pop()
#                            if mat_boundary[p // n, p % n] == 0:
#                                reachable.add(p)
                    else:
                        neighbors = [((row + 1 + n) % n) * n + col, ((row - 1 + n) % n) * n + col, row * n + (col + 1 + n) % n, row * n + (col - 1 + n) % n, ((row - 1 + n) % n) * n + (col - 1 + n) % n, ((row - 1 + n) % n) * n + (col + 1 + n) % n, ((row + 1 + n) % n) * n + (col - 1 + n) % n, ((row + 1 + n) % n) * n + (col + 1 + n) % n]
                        for neighbor in neighbors:
                            if neighbor not in visited_curr:
                                r = neighbor // n
                                c = neighbor % n
                                if mat[r, c] == 0 and mat_boundary[r, c] == 0: #and not non_cooperative) or (mat[r, c] < bar and non_cooperative):
                                    stack.append(neighbor)
                                    visited_curr.add(neighbor)
                for point in visited_curr:
                    visited.add(point)
    return mat_boundary_external

def simulation(Graph, p=0.6, T=10, m=7):
    Locations, Edges, _ = Graph
    q = Queue()
    POPULATION = len(Locations)
    Infected_Dates = np.zeros(POPULATION)
    Infected_Dates[0] = 1
    
    infected_set = set([0])
    
    for _ in range(T):
#        for i in range(POPULATION):
#            if Infected_Dates[i] > 0 and Infected_Dates[i] <= m:
#                q.put(i)
        tmp_set = set([])
        for i in infected_set:
            if Infected_Dates[i] > 0 and Infected_Dates[i] <= m:
                q.put(i)
                tmp_set.add(i)
        infected_set = tmp_set
        while not q.empty():
            point = q.get()
            if Infected_Dates[point] > 0 and Infected_Dates[point] <= m:
                for neighbor in Edges[point]:
                    if Infected_Dates[neighbor] == 0 and np.random.uniform() < p:
                        Infected_Dates[neighbor] = 1
                        infected_set.add(neighbor)
            Infected_Dates[point] += 1
    Is_Infected = (Infected_Dates > 0) + 0
    return Is_Infected, None

def is_boundary_picky(neighbor, non_cooperative, mat_pred, mat, Grid_locs, bar=0.1, max_dist=3, min_coop=10):
    s = 0#np.zeros(4)
    total = 0#np.zeros(4)
    dist = 0
    n = GRID_SIZE
#    row = neighbor // n
#    col = neighbor % n
#    if (row, col) not in Grid_locs or len(Grid_locs[(row, col)]) == 0:
#        return False, False
    while total < min_coop and dist < max_dist:
        dist += 1
        points_arr = []
        if dist == 1:
            points_arr.append((0, 0))
        for i in range(-dist, dist + 1):
            points_arr.append((-dist, i))
            points_arr.append((dist, i))
        for i in range(-dist + 1, dist):
            points_arr.append((i, -dist))
            points_arr.append((i, dist))
#    for a1 in range(-dist, dist + 1):
#        for a2 in range(-dist, dist + 1):
        for point in points_arr:
            a1, a2 = point
            r = (neighbor // n + a1 + n) % n
            c = (neighbor % n + a2 + n) % n
            #pos = 2 * (r > 0) + (c > 0)
            if non_cooperative[r, c] == 0:
                if (r, c) in Grid_locs and len(Grid_locs[(r, c)]) > 0:
                    total += 1#[pos] += 1
                    if mat_pred[r, c] == 0:
                        mat_pred[r, c] = int((mat[r, c] - 0.5) * 2)
                    if mat_pred[r, c] == 1:
                        s += 1#[pos] += 1
    return s / total <= 0.5 + bar and s / total >= 0.5 - bar, s / total < 0.5

def BFS_(mat, grid_population_cooperative, pop, alpha = 0.05, non_cooperative=False, bar=0.1):
    mat_pred = np.zeros(mat.shape)
    center = mat.shape[0] // 2
    n = mat.shape[0]
    idx = center
    total_test_cnt = 0
    mat_boundary = np.zeros((n, n))
    mat_boundary_init = np.zeros((n, n))
    visited = set()
    q = Queue()
    stop_criteria = int(math.ceil(np.log(1/alpha) / (pop / n ** 2)))
    cnt = 0
    while idx < mat.shape[0]:
        if not non_cooperative:
            isBoundary = get_boundary_single(mat, idx, center, mat_pred)
            isInfected = mat[idx, center] > 0
        else:
            isBoundary = get_boundary_single(mat, idx, center, mat_pred, bar)
            isInfected = mat[idx, center] > bar
        if isBoundary and not isInfected:
            mat_boundary_init[idx, center] = 1
            cnt = 0
        elif not isInfected:
            cnt += 1
        else:
            cnt = 0
        if cnt >= stop_criteria:
            break
        #visited.add(idx * n + center)
        idx += 1
    idx = n - 1 - np.argmax(mat_boundary_init[::-1, center])
    q.put(idx * n + center)
    mat_boundary[idx, center] = 1
    while not q.empty():
        point = q.get()
        row = point // n
        col = point % n
        if mat_boundary[row, col] == 1:
            neighbors = [((row + 1 + n) % n) * n + col, ((row - 1 + n) % n) * n + col, row * n + (col + 1 + n) % n, row * n + (col - 1 + n) % n, ((row - 1 + n) % n) * n + (col - 1 + n) % n, ((row - 1 + n) % n) * n + (col + 1 + n) % n, ((row + 1 + n) % n) * n + (col - 1 + n) % n, ((row + 1 + n) % n) * n + (col + 1 + n) % n]
            for neighbor in neighbors:
                if neighbor not in visited:
                    if not non_cooperative:
                        isBoundary = get_boundary_single(mat, neighbor // n, neighbor % n, mat_pred)
                        isInfected = mat[neighbor // n, neighbor % n] > 0
                    else:
                        isBoundary = get_boundary_single(mat, neighbor // n, neighbor % n, mat_pred, bar)
                        isInfected = mat[neighbor // n, neighbor % n] > bar
                    if isBoundary and not isInfected:
                        q.put(neighbor)
                        visited.add(neighbor)
                        mat_boundary[neighbor // n, neighbor % n] = 1
        
    return mat_pred, mat_boundary, np.sum(mat_pred != 0), np.sum(np.abs(mat_pred) * grid_population_cooperative)

def DFS_comparator(n, center, point):
    row = np.abs(point // n - center)
    col = np.abs(point % n - center)
    if row == 0:
        ratio = 2 * n - col
    else:
        ratio = col / row
    return row + col, row ** 2 + col ** 2#row + col, -ratio

def p(row, col, n):
    return (row + n) % n * n + (col + n) % n

def DFS_get_neighbors(n, center, point):
    row = point // n
    col = point % n
    row_dist = np.abs(row - center)
    col_dist = np.abs(col - center)
    if row > center and col >= center and row_dist >= col_dist:
        neighbors = [p(row - 1, col - 1, n), p(row - 1, col, n), p(row - 1, col + 1, n), p(row, col + 1, n), p(row + 1, col + 1, n), p(row + 1, col, n)]#, p(row + 1, col - 1, n), p(row, col - 1, n)]
    elif row >= center and col > center and row_dist < col_dist:
        neighbors = [p(row, col - 1, n), p(row - 1, col - 1, n), p(row - 1, col, n), p(row - 1, col + 1, n), p(row, col + 1, n), p(row + 1, col + 1, n)]#, p(row + 1, col, n), p(row + 1, col - 1, n)]
    elif row < center and col > center and row_dist < col_dist:
        neighbors = [p(row + 1, col - 1, n), p(row, col - 1, n), p(row - 1, col - 1, n), p(row - 1, col, n), p(row - 1, col + 1, n), p(row, col + 1, n)]#, p(row + 1, col + 1, n), p(row + 1, col, n)]
    elif row < center and col >= center and row_dist >= col_dist:
        neighbors = [p(row + 1, col, n), p(row + 1, col - 1, n), p(row, col - 1, n), p(row - 1, col - 1, n), p(row - 1, col, n), p(row - 1, col + 1, n)]#, p(row, col + 1, n), p(row + 1, col + 1, n)]
    elif row < center and col < center and row_dist > col_dist:
        neighbors = [p(row + 1, col + 1, n), p(row + 1, col, n), p(row + 1, col - 1, n), p(row, col - 1, n), p(row - 1, col - 1, n), p(row - 1, col, n)]#, p(row - 1, col + 1, n), p(row, col + 1, n)]
    elif row <= center and col < center and row_dist <= col_dist:
        neighbors = [p(row, col + 1, n), p(row + 1, col + 1, n), p(row + 1, col, n), p(row + 1, col - 1, n), p(row, col - 1, n), p(row - 1, col - 1, n)]#, p(row - 1, col, n), p(row - 1, col + 1, n)]
    elif row > center and col < center and row_dist <= col_dist:
        neighbors = [p(row - 1, col + 1, n), p(row, col + 1, n), p(row + 1, col + 1, n), p(row + 1, col, n), p(row + 1, col - 1, n), p(row, col - 1, n)]#, p(row - 1, col - 1, n), p(row - 1, col, n)]
    else:
        neighbors = [p(row - 1, col, n), p(row - 1, col + 1, n), p(row, col + 1, n), p(row + 1, col + 1, n), p(row + 1, col, n), p(row + 1, col - 1, n)]#, p(row, col - 1, n), p(row - 1, col - 1, n)]
    return neighbors

def DFS_(mat, grid_population_cooperative, pop, alpha = 0.05, non_cooperative=False, bar=0.1):
    mat_pred = np.zeros(mat.shape)
    center = mat.shape[0] // 2
    n = mat.shape[0]
    idx = center
    total_test_cnt = 0
    mat_boundary = np.zeros((n, n))
    mat_boundary_init = np.zeros((n, n))
    visited = set()
    stack = []
    stop_criteria = int(math.ceil(np.log(1/alpha) / (pop / n ** 2)))
    cnt = 0
    while idx < mat.shape[0]:
        if not non_cooperative:
            isBoundary = get_boundary_single(mat, idx, center, mat_pred)
            isInfected = mat[idx, center] > 0
        else:
            isBoundary = get_boundary_single(mat, idx, center, mat_pred, bar)
            isInfected = mat[idx, center] > bar
        if isBoundary and not isInfected:
            mat_boundary_init[idx, center] = 1
            cnt = 0
        elif not isInfected:
            cnt += 1
        else:
            cnt = 0
        if cnt >= stop_criteria:
            break
        #visited.add(idx * n + center)
        idx += 1
    idx = n - 1 - np.argmax(mat_boundary_init[::-1, center])
    mat_boundary[idx, center] = 1
    visited.add(idx * n + center)
    neighbors = [((idx - 1 + n) % n) * n + center, ((idx + 1 + n) % n) * n + center, idx * n + (center + 1 + n) % n]#[((idx - 1 + n) % n) * n + center, ((idx - 1 + n) % n) * n + (center + 1 + n) % n, idx * n + (center + 1 + n) % n, ((idx + 1 + n) % n) * n + center, ((idx + 1 + n) % n) * n + (center + 1 + n) % n]
    #neighbors = sorted(neighbors, key = lambda x: (np.abs(x // n - center) + np.abs(x % n - center), (x // n - center) ** 2 + (x % n - center) ** 2))
    #neighbors = sorted(neighbors, key = lambda x: (x // n - center) ** 2 + (x % n - center) ** 2)
    neighbors = sorted(neighbors, key = lambda x: DFS_comparator(n, center, x))
    for neighbor in neighbors:
        if not non_cooperative:
            isBoundary = get_boundary_single(mat, neighbor // n, neighbor % n, mat_pred)
            isInfected = mat[neighbor // n, neighbor % n] > 0
        else:
            isBoundary = get_boundary_single(mat, neighbor // n, neighbor % n, mat_pred, bar)
            isInfected = mat[neighbor // n, neighbor % n] > bar
        if isBoundary and not isInfected:
            stack.append(neighbor)
            visited.add(neighbor)
            mat_boundary[neighbor // n, neighbor % n] = 1
    quit = False
    
    while len(stack) > 0 and not quit:
        point = stack.pop()
        row = point // n
        col = point % n
        if mat_boundary[row, col] == 1:
            neighbors = [((row + 1 + n) % n) * n + col, ((row - 1 + n) % n) * n + col, row * n + (col + 1 + n) % n, row * n + (col - 1 + n) % n]#DFS_get_neighbors(n, center, point)#[((row + 1 + n) % n) * n + col, ((row - 1 + n) % n) * n + col, row * n + (col + 1 + n) % n, row * n + (col - 1 + n) % n, ((row - 1 + n) % n) * n + (col - 1 + n) % n, ((row - 1 + n) % n) * n + (col + 1 + n) % n, ((row + 1 + n) % n) * n + (col - 1 + n) % n, ((row + 1 + n) % n) * n + (col + 1 + n) % n]
            #neighbors = sorted(neighbors, key = lambda x: (np.abs(x // n - center) + np.abs(x % n - center), (x // n - center) ** 2 + (x % n - center) ** 2))
            neighbors = sorted(neighbors, key = lambda x: DFS_comparator(n, center, x))
            
            for neighbor in neighbors:
                if col == center - 1 and neighbor // n <= idx + 1 and neighbor // n >= idx - 1 and neighbor % n == center:
                    if not non_cooperative:
                        isInfected = mat[neighbor // n, neighbor % n] > 0
                    else:
                        isInfected = mat[neighbor // n, neighbor % n] > bar
                    if mat_boundary[neighbor // n, center] and not isInfected:
                        quit = True
                        #print("quit!")
                        break
                if neighbor not in visited:
                    if not non_cooperative:
                        isBoundary = get_boundary_single(mat, neighbor // n, neighbor % n, mat_pred)
                        isInfected = mat[neighbor // n, neighbor % n] > 0
                    else:
                        isBoundary = get_boundary_single(mat, neighbor // n, neighbor % n, mat_pred, bar)
                        isInfected = mat[neighbor // n, neighbor % n] > bar
                    if isBoundary and not isInfected:
                        stack.append(neighbor)
                        visited.add(neighbor)
                        mat_boundary[neighbor // n, neighbor % n] = 1

    return mat_pred, mat_boundary, np.sum(mat_pred != 0), np.sum(np.abs(mat_pred) * grid_population_cooperative)

def SimilarDistance(mat, grid_population_cooperative, Grid_locs, non_cooperative=False, bar=0.1):
    mat_pred = np.zeros(mat.shape)
    center = mat.shape[0] // 2
    n = mat.shape[0]
    idx = center
    total_test_cnt = 0
    mat_boundary = np.zeros((n, n))
    visited = set()
    q = Queue()
    while idx < mat.shape[0]:
        if not non_cooperative:
            get_boundary_single(mat, idx, center, mat_pred)
        else:
            get_boundary_single(mat, idx, center, mat_pred, bar)
        #visited.add(idx * n + center)
        idx += 1
    idx = np.argmax(mat[:, center])
    mat_boundary[idx, center] = 1
    dist = np.abs(idx - center)
    
    for i in range(-dist, dist + 1):
        row = (center + i) % n
        col1 = (center + dist - abs(i)) % n
        col2 = (center - dist + abs(i)) % n
        if row * n + col1 not in visited:
            q.put(row * n + col1)
        if row * n + col2 not in visited:
            q.put(row * n + col2)

    while not q.empty():
        point = q.get()
        row = point // n
        col = point % n
        if not non_cooperative:
            isBoundary = get_boundary_single(mat, row, col, mat_pred)
            isInfected = mat[row, col] > 0
        else:
            isBoundary = get_boundary_single(mat, row, col, mat_pred, bar)
            isInfected = mat[row, col] > bar
        if isBoundary:
            mat_boundary[row, col] = 1
        dist_point = np.abs(row - center) + np.abs(col - center)
        neighbors = [((row + 1 + n) % n) * n + col, ((row - 1 + n) % n) * n + col, row * n + (col + 1 + n) % n, row * n + (col - 1 + n) % n, ((row - 1 + n) % n) * n + (col - 1 + n) % n, ((row - 1 + n) % n) * n + (col + 1 + n) % n, ((row + 1 + n) % n) * n + (col - 1 + n) % n, ((row + 1 + n) % n) * n + (col + 1 + n) % n]
        for neighbor in neighbors:
            if neighbor not in visited:
                dist_neighbor = np.abs(neighbor // n - center) + np.abs(neighbor % n - center)
                if Grid_locs[row, col] == 0 or (not isInfected and dist_neighbor <= dist_point) or (isInfected and dist_neighbor >= dist_point):
                    if not isBoundary:
                        q.put(neighbor)
                    else:
                        if not non_cooperative:
                            isBoundary_ = get_boundary_single(mat, neighbor // n, neighbor % n, mat_pred)
                        else:
                            isBoundary_ = get_boundary_single(mat, neighbor // n, neighbor % n, mat_pred, bar)
                        if isBoundary_:
                            mat_boundary[neighbor // n, neighbor % n] = 1
                    visited.add(neighbor)
    
    return mat_pred, mat_boundary, np.sum(mat_pred != 0), np.sum(np.abs(mat_pred) * grid_population_cooperative)

def fill_infected_region(mat_boundary_external):
    n = mat_boundary_external.shape[0]
    center = n // 2
    mat_filled = np.zeros((n, n))
    q = Queue()
    q.put(center * n + center)
    while not q.empty():
        point = q.get()
        row = point // n
        col = point % n
        mat_filled[row, col] = 1
        neighbors = [((row + 1 + n) % n) * n + col, ((row - 1 + n) % n) * n + col, row * n + (col + 1 + n) % n, row * n + (col - 1 + n) % n, ((row - 1 + n) % n) * n + (col - 1 + n) % n, ((row - 1 + n) % n) * n + (col + 1 + n) % n, ((row + 1 + n) % n) * n + (col - 1 + n) % n, ((row + 1 + n) % n) * n + (col + 1 + n) % n]#[((row + 1 + n) % n) * n + col, ((row - 1 + n) % n) * n + col, row * n + (col + 1 + n) % n, row * n + (col - 1 + n) % n]
        for neighbor in neighbors:
            r = neighbor // n
            c = neighbor % n
            if mat_filled[r, c] == 0 and mat_boundary_external[r, c] == 0:
                q.put(neighbor)
                mat_filled[r, c] = 1
    return mat_filled

def shrink_grid_map(N, radius=0):
    center = N // 2
    r = (center - radius) % (2 * radius + 1)
    n = int(math.ceil((center - radius) / (2 * radius + 1)) + 1 + math.ceil((N - center - radius - 1) / (2 * radius + 1)))
    map_lst = []
    for i in range(n):
        if i == 0:
            lo = 0
            hi = r
        else:
            lo = r + (2 * radius + 1) * i
            hi = min(r + (2 * radius + 1) * (i + 1), N)
        map_lst.append((lo, hi))
    return map_lst

def shrink_grid(orig_grid, shrink_map):
    n = len(shrink_map)
    shrinked_grid = np.zeros((n, n))
    for i in range(n):
        tup_i = shrink_map[i]
        for j in range(n):
            tup_j = shrink_map[j]
            shrinked_grid[i, j] = np.sum(orig_grid[tup_i[0]:tup_i[1], tup_j[0]:tup_j[1]])
    return shrinked_grid

def expand_grid(shrinked_grid, shrink_map, N):
    expanded_grid = np.zeros((N, N))
    n = len(shrink_map)
    for i in range(n):
        tup_i = shrink_map[i]
        for j in range(n):
            tup_j = shrink_map[j]
            expanded_grid[tup_i[0]:tup_i[1], tup_j[0]:tup_j[1]] = shrinked_grid[i, j]
    return expanded_grid

def plot_matrices(orig_mat, boundary_mat, mat_pred_bfs, mat_pred_dfs, test_cnt_bfs, test_cnt_dfs, bar, non_coop, benchmark):
    false_pos_bfs = np.sum(boundary_mat < mat_pred_bfs)
    false_neg_bfs = np.sum(boundary_mat > mat_pred_bfs)
    false_pos_dfs = np.sum(boundary_mat < mat_pred_dfs)
    false_neg_dfs = np.sum(boundary_mat > mat_pred_dfs)
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax[0, 0].imshow(orig_mat, cmap="hot", interpolation="nearest")
    ax[0, 1].imshow(boundary_mat, cmap="hot", interpolation="nearest")
    ax[1, 0].imshow(mat_pred_bfs, cmap="hot", interpolation="nearest")
    ax[1, 1].imshow(mat_pred_dfs, cmap="hot", interpolation="nearest")
    ax[0, 0].set_title("Infection Status\n# Infected People = " + str(np.sum(orig_mat)))
    ax[0, 1].set_title("True Boundary\nTrue # Boundary People = " + str(np.sum(boundary_mat)))
    ax[1, 0].set_title("Predicted Boundary BFS\nPredicted # Boundary People = " + str(np.sum(mat_pred_bfs)) + "\nFalsePos = " + str(false_pos_bfs) + " FalseNeg = " + str(false_neg_bfs) + "\nTotal Test = " + str(test_cnt_bfs))
    ax[1, 1].set_title("Predicted Boundary DFS\nPredicted # Boundary People = " + str(np.sum(mat_pred_dfs)) + "\nFalsePos = " + str(false_pos_dfs) + " FalseNeg = " + str(false_neg_dfs) + "\nTotal Test = " + str(test_cnt_dfs))
    plt.suptitle("Bar = " + str(bar) + " Non-Cooperative Rate = " + str(non_coop) + "\n Internal Cells UNION External Boundary = " + str(int(benchmark)))
    plt.savefig(f"GeneralDebug_bar={bar}_non-coop={non_coop}.png")
    plt.clf()

def add_mat(mat_pred, mat):
    ret = mat_pred * (-2) + (1 - mat_pred) * mat
    ret[ret == 1] = 2
    ret[ret == -1] = 3
    ret[ret == -2] = 1
    return ret

def plot_matrices2(orig_mat, boundary_mat, mat_pred_bfs, mat_pred_dfs, test_cnt_bfs, test_cnt_dfs, bar, non_coop, benchmark, p, radius, pop, T, infected_coverage_people_bfs, infected_coverage_people_dfs, susceptible_coverage_people_bfs, susceptible_coverage_people_dfs, mat_bfs, mat_dfs):
    black_patch = mpatches.Patch(color = "black", label = "Not Tested")
    blue_patch = mpatches.Patch(color = "blue", label = "Infected")
    white_patch = mpatches.Patch(color = "white", label = "Predicted External Boundary")
    red_patch = mpatches.Patch(color = "red", label = "Non-Infected")
    legend_pred = [black_patch, blue_patch, white_patch, red_patch]
    
    legend_infect = [mpatches.Patch(color = "red", label = "Non-Infected"), mpatches.Patch(color = "blue", label = "Infected")]
    legend_boundary = [mpatches.Patch(color = "darkgreen", label = "Not External Boundary"), mpatches.Patch(color = "white", label = "External Boundary")]

    false_pos_bfs = np.sum(boundary_mat < mat_pred_bfs)
    false_neg_bfs = np.sum(boundary_mat > mat_pred_bfs)
    false_pos_dfs = np.sum(boundary_mat < mat_pred_dfs)
    false_neg_dfs = np.sum(boundary_mat > mat_pred_dfs)

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    cmap = colors.ListedColormap(['red', 'blue'])
    bounds=[0,0.5,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    #ax[0, 0].imshow(orig_mat, cmap="hot", interpolation="nearest")
    ax[0, 0].imshow(orig_mat, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
    ax[0, 0].legend(handles=legend_infect)
    
    cmap = colors.ListedColormap(['darkgreen', 'white'])
    bounds=[0,0.5,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax[0, 1].imshow(boundary_mat, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
    ax[0, 1].legend(handles=legend_boundary)

    cmap = colors.ListedColormap(['black', 'white', 'blue', 'red'])
    bounds=[0, 0.5, 1, 1.5, 2, 2.5, 3]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    #ax[1, 0].imshow(add_mat(mat_pred_bfs, mat_bfs), cmap="hot", interpolation="nearest")
    ax[1, 0].imshow(add_mat(mat_pred_bfs, mat_bfs), interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
    ax[1, 0].legend(handles=legend_pred)
    
    cmap = colors.ListedColormap(['black', 'white', 'blue', 'red'])
    bounds=[0, 0.5, 1, 1.5, 2, 2.5, 3]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    #ax[1, 1].imshow(add_mat(mat_pred_dfs, mat_dfs), cmap="hot", interpolation="nearest")
    ax[1, 1].imshow(add_mat(mat_pred_dfs, mat_dfs), interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
    ax[1, 1].legend(handles=legend_pred)
    ax[0, 0].set_title("Infection Status\n# Infected People = " + str(np.sum(orig_mat)))
    ax[0, 1].set_title("True Boundary\nTrue # Boundary People = " + str(np.sum(boundary_mat)))
#    ax[1, 0].set_title("Predicted Boundary BFS\nPredicted # Boundary People = " + str(np.sum(mat_pred_bfs)) + "\nFalsePos = " + str(false_pos_bfs) + " FalseNeg = " + str(false_neg_bfs) + "\nInfectedCoverageRate = " + str(round(infected_coverage_people_bfs, 2)) + " SusceptibleCoverage = " + str(round(susceptible_coverage_people_bfs, 2)) + "\nTotal Test = " + str(test_cnt_bfs))
#    ax[1, 1].set_title("Predicted Boundary DFS\nPredicted # Boundary People = " + str(np.sum(mat_pred_dfs)) + "\nFalsePos = " + str(false_pos_dfs) + " FalseNeg = " + str(false_neg_dfs) + "\nInfectedCoverageRate = " + str(round(infected_coverage_people_dfs, 2)) + " SusceptibleCoverage = " + str(round(susceptible_coverage_people_dfs, 2)) + "\nTotal Test = " + str(test_cnt_dfs))
#    plt.suptitle("p = " + str(p) + " radius = " + str(radius) + " T = " + str(T) + " population = " + str(pop) + "\nBar = " + str(bar) + " Non-Cooperative Rate = " + str(non_coop) + "\nInternal Cells UNION External Boundary = " + str(int(benchmark)))
    if non_coop > 0:
        ax[1, 0].set_title("Predicted Boundary -- Full Search" + "\nMisclassified As Boundary = " + str(round(susceptible_coverage_people_bfs / benchmark, 2)) + " Misclassified Infected People = " + str(round(infected_coverage_people_bfs / benchmark, 2)) + "\nTotal Test / N = " + str(round(test_cnt_bfs / benchmark, 2)))
        ax[1, 1].set_title("Predicted Boundary -- Geometric Search\nMisclassified As Boundary = " + str(round(susceptible_coverage_people_dfs / benchmark, 2)) + " Misclassified Infected People = " + str(round(infected_coverage_people_dfs / benchmark, 2)) + "\nTotal Test / N = " + str(round(test_cnt_dfs / benchmark, 2)))
        plt.suptitle("Algorithm Result (Non-Cooperative)")
    else:
        ax[1, 0].set_title("Predicted Boundary -- Full Search\nMisclassified As Boundary = " + str(round(false_pos_bfs / benchmark, 2)) + " Misclassified Infected People = " + str(round(infected_coverage_people_bfs / benchmark, 2)) + "\nTotal Test / N = " + str(round(test_cnt_bfs / benchmark, 2)))
        ax[1, 1].set_title("Predicted Boundary -- Geometric Search\nMisclassified As Boundary = " + str(round(false_pos_dfs / benchmark, 2)) + " Misclassified Infected People = " + str(round(infected_coverage_people_dfs / benchmark, 2)) + "\nTotal Test / N = " + str(round(test_cnt_dfs / benchmark, 2)))
        plt.suptitle("Algorithm Result (Fully Cooperative)")
    plt.subplots_adjust(hspace=0.25)
    plt.savefig(f"../Plots/General_bar={bar}_non-coop={non_coop}_p={p}_radius={radius}_pop={pop}_T={T}.tiff", dpi=300)
    plt.clf()

def run_exp(exp_lst, exp_lst2):
    dct = {"p":[], "T":[], "NonCooperativeProb":[], "Bar":[], "ExperimentNumber":[], "Population":[], "Algorithm":[], "Radius":[], "ExternalBoundaryCell":[], "ExternalBoundaryPeople":[], "InternalCell":[], "InternalPeople":[], "FalsePosCell":[], "FalsePosPeople":[], "FalseNegCell":[], "FalseNegPeople":[], "TotalTestCell":[], "TotalTestPeople":[], "InfectedCoverageCell":[], "InfectedCoveragePeople":[], "SusceptibleCoverageCell":[], "SusceptibleCoveragePeople":[]}
    for tup in tqdm(exp_lst):
        p, T, exp_num, pop = tup
        G = sample_graph(pop)
        Locations, Edges, Grid_locs = G
        n = GRID_SIZE
        Is_Infected, Is_Boundary = simulation(G, p = p, T = T, m = 7)
        for tup3 in exp_lst2:
            algo, tup2 = tup3
    #        if non_cooperative_prob > 0:
    #            exp_lst2_curr = exp_lst2
    #        else:
    #            exp_lst2_curr = [(0, 0)]
    #
    #        for tup2 in exp_lst2_curr:
            non_cooperative_prob, bar, radius = tup2
            bar = bar * p
#            G = sample_graph(pop)
#            Locations, Edges, Grid_locs = G
#            n = GRID_SIZE
            
            IsCooperative = np.random.uniform(size=pop)
            IsCooperative[IsCooperative <= non_cooperative_prob] = 0
            IsCooperative[IsCooperative > 0] = 1
#            Is_Infected, Is_Boundary = simulation(G, p = p, T = T, m = 7)
            grid_infected = np.zeros((GRID_SIZE, GRID_SIZE))
            grid_population = np.zeros((n, n))
            grid_population_cooperative = np.zeros((n, n))
            grid_infected_cooperative = np.zeros((n, n))
            for i in range(pop):
                horizontal, vertical = Locations[i]
                grid_infected[int(vertical), int(horizontal)] += Is_Infected[i]
                grid_population[int(vertical), int(horizontal)] += 1
                grid_population_cooperative[int(vertical), int(horizontal)] += IsCooperative[i]
                grid_infected_cooperative[int(vertical), int(horizontal)] += IsCooperative[i] * Is_Infected[i]

            mat_boundary = get_boundary(grid_infected)
            mat0 = grid_infected
            mat0[mat0 > 0] = 1
            mat_boundary_external = get_boundary_external(mat0, mat_boundary)
            mat_filled = fill_infected_region(mat_boundary_external)
            
            if non_cooperative_prob > 0 and radius > 0:
                shrink_map = shrink_grid_map(n, radius)
                grid_infected_cooperative_shrinked = shrink_grid(grid_infected_cooperative, shrink_map)
                grid_population_cooperative_shrinked = shrink_grid(grid_population_cooperative, shrink_map)
                mat = grid_infected_cooperative_shrinked / grid_population_cooperative_shrinked
                mat[np.isnan(mat)] = 0
                if np.sum(np.isinf(mat)) > 0:
                    raise Exception("Invalid Division Occurred!")
            else:
                mat = mat0
                grid_population_cooperative_shrinked = grid_population_cooperative
            
            external_boundary_cell = np.sum(mat_boundary_external)
            external_boundary_people = np.sum(mat_boundary_external * grid_population)
            internal_cell = np.sum(mat_filled)
            internal_people = np.sum(mat_filled * grid_population)
                            
            if algo == "BFS":
                _, mat_boundary_pred_shrinked, total_test_cell, total_test_people = BFS_(mat, grid_population_cooperative_shrinked, pop, non_cooperative=non_cooperative_prob>0, bar=bar)
            else:
                _, mat_boundary_pred_shrinked, total_test_cell, total_test_people = DFS_(mat, grid_population_cooperative_shrinked, pop, non_cooperative=non_cooperative_prob>0, bar=bar)
            
            if non_cooperative_prob > 0 and radius > 0:
                mat_boundary_pred = expand_grid(mat_boundary_pred_shrinked, shrink_map, n)
            else:
                mat_boundary_pred = mat_boundary_pred_shrinked
            
            mat_filled_predict = fill_infected_region(mat_boundary_pred)
            susceptible_coverage_cell = np.sum(mat_filled_predict > mat_filled)
            susceptible_coverage_people = np.sum((mat_filled_predict > mat_filled) * (grid_population - grid_infected))
            mat_filled_predict += mat_boundary_pred
            mat_filled_predict[mat_filled_predict > 0] = 1
            infected_coverage_cell = np.sum(mat_filled_predict * mat0) / np.sum(mat_filled * mat0)
            infected_coverage_people = np.sum(mat_filled_predict * grid_infected) / np.sum(mat_filled * grid_infected)
            
            false_pos_cell = np.sum(mat_boundary_external < mat_boundary_pred)
            false_pos_people = np.sum((mat_boundary_external < mat_boundary_pred) * grid_population)
            false_neg_cell = np.sum(mat_boundary_external > mat_boundary_pred)
            false_neg_people = np.sum((mat_boundary_external > mat_boundary_pred) * grid_population)

            dct["p"].append(p)
            dct["T"].append(T)
            dct["NonCooperativeProb"].append(non_cooperative_prob)
            dct["Bar"].append(bar)
            dct["ExperimentNumber"].append(exp_num + 1)
            dct["Population"].append(pop)
            dct["Algorithm"].append(algo)
            dct["Radius"].append(radius)
            dct["ExternalBoundaryCell"].append(external_boundary_cell)
            dct["ExternalBoundaryPeople"].append(external_boundary_people)
            dct["InternalCell"].append(internal_cell)
            dct["InternalPeople"].append(internal_people)
            dct["FalsePosCell"].append(false_pos_cell)
            dct["FalsePosPeople"].append(false_pos_people)
            dct["FalseNegCell"].append(false_neg_cell)
            dct["FalseNegPeople"].append(false_neg_people)
            dct["TotalTestCell"].append(total_test_cell)
            dct["TotalTestPeople"].append(total_test_people)
            dct["InfectedCoverageCell"].append(infected_coverage_cell)
            dct["InfectedCoveragePeople"].append(infected_coverage_people)
            dct["SusceptibleCoverageCell"].append(susceptible_coverage_cell)
            dct["SusceptibleCoveragePeople"].append(susceptible_coverage_people)
    return dct

def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
    return newfunc

@timeit
def single_exp(p, non_cooperative_prob, radius, pop, T):
    G = sample_graph(pop)
    Locations, Edges, Grid_locs = G
    bar = p / 2
    n = GRID_SIZE
    IsCooperative = np.random.uniform(size=pop)
    IsCooperative[IsCooperative <= non_cooperative_prob] = 0
    IsCooperative[IsCooperative > 0] = 1
    Is_Infected, Is_Boundary = simulation(G, p = p, T = T, m = 7)

    grid_infected = np.zeros((n, n))
    grid_population = np.zeros((n, n))
    grid_population_cooperative = np.zeros((n, n))
    grid_infected_cooperative = np.zeros((n, n))
    for i in range(pop):
        horizontal, vertical = Locations[i]
        grid_infected[int(vertical), int(horizontal)] += Is_Infected[i]
        grid_population[int(vertical), int(horizontal)] += 1
        grid_population_cooperative[int(vertical), int(horizontal)] += IsCooperative[i]
        grid_infected_cooperative[int(vertical), int(horizontal)] += IsCooperative[i] * Is_Infected[i]

    mat0 = grid_infected
    mat0[mat0 > 0] = 1
    mat_boundary = get_boundary(grid_infected)
    mat_boundary_external = get_boundary_external(mat0, mat_boundary)
    mat_filled = fill_infected_region(mat_boundary_external)
    benchmark = np.sum(mat_filled) + np.sum(mat_boundary_external)
    benchmark_people = np.sum((mat_filled + mat_boundary_external) * grid_population)

    if non_cooperative_prob > 0 and radius > 0:
        shrink_map = shrink_grid_map(n, radius)
        grid_infected_cooperative_shrinked = shrink_grid(grid_infected_cooperative, shrink_map)
        grid_population_cooperative_shrinked = shrink_grid(grid_population_cooperative, shrink_map)
        mat = grid_infected_cooperative_shrinked / grid_population_cooperative_shrinked
        mat[np.isnan(mat)] = 0
        if np.sum(np.isinf(mat)) > 0:
            raise Exception("Invalid Division Occurred!")
    else:
        mat = mat0
        grid_population_cooperative_shrinked = grid_population_cooperative

    external_boundary_cell = np.sum(mat_boundary_external)
    external_boundary_people = np.sum(mat_boundary_external * grid_population)
    internal_cell = np.sum(mat_filled)
    internal_people = np.sum(mat_filled * grid_population)

    mat_bfs, mat_boundary_pred_shrinked_bfs, total_test_cell_bfs, total_test_people_bfs = BFS_(mat, grid_population_cooperative_shrinked, POPULATION, non_cooperative=non_cooperative_prob>0, bar=bar)
    mat_dfs, mat_boundary_pred_shrinked_dfs, total_test_cell_dfs, total_test_people_dfs = DFS_(mat, grid_population_cooperative_shrinked, POPULATION, non_cooperative=non_cooperative_prob>0, bar=bar)

    if non_cooperative_prob > 0 and radius > 0:
        mat_boundary_pred_bfs = expand_grid(mat_boundary_pred_shrinked_bfs, shrink_map, n)
        mat_boundary_pred_dfs = expand_grid(mat_boundary_pred_shrinked_dfs, shrink_map, n)
        mat_bfs = expand_grid(mat_bfs, shrink_map, n)
        mat_dfs = expand_grid(mat_dfs, shrink_map, n)
    else:
        mat_boundary_pred_bfs = mat_boundary_pred_shrinked_bfs
        mat_boundary_pred_dfs = mat_boundary_pred_shrinked_dfs

    mat_filled_predict_bfs = fill_infected_region(mat_boundary_pred_bfs)
    susceptible_coverage_cell_bfs = np.sum(mat_filled_predict_bfs > mat_filled)
    susceptible_coverage_people_bfs = np.sum((mat_filled_predict_bfs > mat_filled) * (grid_population - grid_infected))
    mat_filled_predict_bfs += mat_boundary_pred_bfs
    mat_filled_predict_bfs[mat_filled_predict_bfs > 0] = 1
    infected_coverage_cell_bfs = np.sum(mat_filled_predict_bfs * mat0) / np.sum(mat_filled * mat0)
    infected_coverage_people_bfs = np.sum(mat_filled_predict_bfs * grid_infected) / np.sum(mat_filled * grid_infected)

    mat_filled_predict_dfs = fill_infected_region(mat_boundary_pred_dfs)
    susceptible_coverage_cell_dfs = np.sum(mat_filled_predict_dfs > mat_filled)
    susceptible_coverage_people_dfs = np.sum((mat_filled_predict_dfs > mat_filled) * (grid_population - grid_infected))
    mat_filled_predict_dfs += mat_boundary_pred_dfs
    mat_filled_predict_dfs[mat_filled_predict_dfs > 0] = 1
    infected_coverage_cell_dfs = np.sum(mat_filled_predict_dfs * mat0) / np.sum(mat_filled * mat0)
    infected_coverage_people_dfs = np.sum(mat_filled_predict_dfs * grid_infected) / np.sum(mat_filled * grid_infected)

    plot_matrices2(mat0, mat_boundary_external, mat_boundary_pred_bfs, mat_boundary_pred_dfs, total_test_people_bfs, total_test_people_dfs, bar, non_cooperative_prob, benchmark_people, p, radius, pop, T, infected_coverage_people_bfs, infected_coverage_people_dfs, susceptible_coverage_people_bfs, susceptible_coverage_people_dfs, mat_bfs, mat_dfs)

def demo(p, T, size=10):
    G = sample_graph(size ** 2, GRID_SIZE = size)
    Locations, Edges, Grid_locs = G
    bar = p / 2
    n = size
    pop = size ** 2
    
    non_cooperative_prob = 0
    IsCooperative = np.random.uniform(size=pop)
    IsCooperative[IsCooperative <= non_cooperative_prob] = 0
    IsCooperative[IsCooperative > 0] = 1
    Is_Infected, Is_Boundary = simulation(G, p = p, T = T, m = 7)

    grid_infected = np.zeros((n, n))
    grid_population = np.zeros((n, n))
    grid_population_cooperative = np.zeros((n, n))
    grid_infected_cooperative = np.zeros((n, n))
    for i in range(pop):
        horizontal, vertical = Locations[i]
        grid_infected[int(vertical), int(horizontal)] += Is_Infected[i]
        grid_population[int(vertical), int(horizontal)] += 1
        grid_population_cooperative[int(vertical), int(horizontal)] += IsCooperative[i]
        grid_infected_cooperative[int(vertical), int(horizontal)] += IsCooperative[i] * Is_Infected[i]

    mat0 = grid_infected
    mat0[mat0 > 0] = 1
    mat_boundary = get_boundary(grid_infected)
    mat_boundary_external = get_boundary_external(mat0, mat_boundary)
    mat_filled = fill_infected_region(mat_boundary_external)

    external_boundary_cell = np.sum(mat_boundary_external)
    external_boundary_people = np.sum(mat_boundary_external * grid_population)
    internal_cell = np.sum(mat_filled)
    internal_people = np.sum(mat_filled * grid_population)
    
    mat_plot = grid_infected + 2 * mat_boundary_external
    #print(Is_Infected)
    
    blue_patch = mpatches.Patch(color = "blue", label = "Infected")
    white_patch = mpatches.Patch(color = "red", label = "External Boundary")
    red_patch = mpatches.Patch(color = "white", label = "Non-Infected")
    legend_pred = [blue_patch, white_patch, red_patch]
    
    cmap = colors.ListedColormap(['white', 'blue', 'red'])
    bounds=[0, 0.5, 1, 1.5, 2]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    #ax[1, 0].imshow(add_mat(mat_pred_bfs, mat_bfs), cmap="hot", interpolation="nearest")
    plt.imshow(mat_plot, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
    plt.axis("off")
    #plt.legend(handles=legend_pred)
    plt.savefig("../Plots/demo.tiff", dpi=300)
    
    #return None

#p_arr = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]#[0.2, 0.4, 0.6, 0.8]
#n = 500
#T_arr = [50, 100, 200]
#non_cooperative_arr = [0.1, 0.333]#[0.05, 0.1, 0.2, 0.4]
#bar_arr = [0, 0.5] #[0, 0.1, 0.2, 0.4]#[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
#exp_num_arr = list(range(20))
#algo_arr = ["BFS", "DFS"]
#radius_arr = [1, 2, 3]
#m = 7
#pop_arr = [500000, 1000000]#[125000, 250000, 500000]
#
#exp_lst2 = [(0, 0, 0)] + list(itertools.product(*[non_cooperative_arr, bar_arr, radius_arr]))
#exp_lst = list(itertools.product(*[p_arr, T_arr, exp_num_arr, pop_arr]))
#exp_lst2 = list(itertools.product(*[algo_arr, exp_lst2]))
#
#print(len(exp_lst))
#batch_size = math.ceil(len(exp_lst) / n_cpu)
#dct_lst = Parallel(n_jobs=n_cpu, backend="multiprocessing")(delayed(run_exp)(
#    exp_lst[i * batch_size : min(len(exp_lst), (i + 1) * batch_size)], exp_lst2
#) for i in range(n_cpu))
#
#dct_all = dct_lst[0]
#for i in range(1, len(dct_lst)):
#    for key in dct_all:
#        dct_all[key] += dct_lst[i][key]
#
#df = pd.DataFrame.from_dict(dct_all)
#df.to_csv("BoundaryResultsGeneral.csv", index=False)
#print("Jobs Done!")

########################

single_exp(p=0.1, non_cooperative_prob=0, radius=1, pop=250000, T=200)
single_exp(p=0.1, non_cooperative_prob=0.333, radius=1, pop=250000, T=200)

########################
#demo(p=0.1, T=180, size=200)
