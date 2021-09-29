import math, sys, json, time, os, os.path
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

#matplotlib.use("Agg")

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

## CONSTANTS
PATH_2_GRAPH = "../Models/graph_"
#PATH_2_TRANSE = "../Models/"
#PATH_2_GCN = "../Models/gcn_"
PATH_2_RESULT = "../Results/"
#PATH2_SIM = "../PandemicSimulations/"
GRAPH_DIR = ""

EMBEDDING_DIM = 50
MARGIN = 10
HIDDEN_DIM_LST = [50, 50, 50, 50]
##

def load_graph(n_nodes, avg_degree, seed=0):
    global GRAPH_DIR
    path_name = PATH_2_GRAPH + "N=" + str(n_nodes) + "_deg=" + str(avg_degree) + "_seed=" + str(seed)
    fname = path_name + "/graph.txt"
    GRAPH_DIR = path_name
    if os.path.exists(path_name):
        print("Graph exists, loading graph...")
        graph = {}
        with open(fname, "r") as f:
            for line in f:
                parent = int(line.split(":")[0])
                child_tups = line.strip("\n").split(":")[1].split(";")
                if len(child_tups) == 1 and child_tups[0].strip() == "":
                    graph[parent] = []
                else:
                    graph[parent] = [(int(x.split(",")[0]), float(x.split(",")[1])) for x in child_tups]
    else:
        print("Graph not found, sampling a graph...")
        np.random.seed(seed)
        degree_lst = np.random.poisson(avg_degree, n_nodes)
        graph = {}
        for i in tqdm(range(n_nodes)):
            neighbors = list(np.random.choice(n_nodes, degree_lst[i], replace=False))
            graph[i] = [(int(x), float(np.random.uniform())) for x in neighbors if x != i]
        print("Done sampling the graph!")
        os.mkdir(path_name)
        os.mkdir(path_name + "/PandemicSimulations/")
        #os.mkdir(path_name + "/TransE/")
        os.mkdir(path_name + "/GCN/")
        with open(fname, "w") as f:
            for i in graph:
                line = str(i) + ":"
                for tup in graph[i]:
                    line += str(tup[0]) + "," + str(tup[1]) + ";"
                line = line.strip(";")
                f.write(line + "\n")
    return graph

def bernoulli(rate):
    return np.random.uniform() < rate

def simulate_pandemic(graph, recover_rate, T, model):
    assert model in ["SIS", "SIR"]
    results = np.zeros(len(graph))
    results[0] = 1
    for _ in range(T):
        infected_individuals = np.where(results == 1)[0]
        for person in infected_individuals:
            recovered = bernoulli(recover_rate)
            if recovered and model == "SIS":
                results[person] = 0
            elif recovered and model == "SIR":
                results[person] = 2
            for neighbor_tup in graph[person]:
                infecting = bernoulli(neighbor_tup[1])
                if infecting:
                    results[neighbor_tup[0]] = 1
    return list(results)

def simulate_multiple(graph, recover_rate, T, model, N):
    path_sim = GRAPH_DIR + "/PandemicSimulations/"
    fname = path_sim + model + "_gamma=" + str(recover_rate) + "_T=" + str(T) + "_N=" + str(N) + ".txt"
    sim_matrix = torch.zeros((N, len(graph)))
    i = 0
    if os.path.exists(fname):
        with open(fname, "r") as f:
            for line in f:
                vec = [int(x) for x in line.strip("\n").split(",")]
                sim_matrix[i,:] = torch.tensor(vec)
                i += 1
    else:
        print("Simulating pandemic paths...")
        with open(fname, "w") as f:
            for i in tqdm(range(N)):
                vec = simulate_pandemic(graph, recover_rate, T, model)
                line = ",".join([str(int(x)) for x in vec])
                f.write(line + "\n")
                sim_matrix[i,:] = torch.tensor(vec)
        print("Done simulation!")
    return sim_matrix
    
def get_transition_matrix(graph):
    mat = torch.zeros((len(graph), len(graph)))
    for i in graph:
        for neighbor_tup in graph[i]:
            mat[i, neighbor_tup[0]] = neighbor_tup[1]
    return mat

class GCN(nn.Module):
    def __init__(self, graph, model):
        super(GCN, self).__init__()
        
        if model == "SIS":
            output_dim = 2
        else:
            output_dim = 3

        self.A_hat = self.get_A_hat(graph)
        
        self.layer_input_emb = nn.GRUCell(output_dim, EMBEDDING_DIM)
        #self.layer_input_emb2 = nn.GRUCell(EMBEDDING_DIM, EMBEDDING_DIM)
#        self.layer_input_gcn = nn.Linear(EMBEDDING_DIM, HIDDEN_DIM)
#        self.layer_1_gcn = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
#        self.layer_2_gcn = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
#        self.layer_output_gcn = nn.Linear(HIDDEN_DIM, output_dim)
        self.layer_lst = nn.ModuleList()
        self.layer_lst.append(nn.Linear(EMBEDDING_DIM, HIDDEN_DIM_LST[0]))
        for i in range(1, len(HIDDEN_DIM_LST)):
            self.layer_lst.append(nn.Linear(HIDDEN_DIM_LST[i - 1], HIDDEN_DIM_LST[i]))
        self.layer_lst.append(nn.Linear(HIDDEN_DIM_LST[-1], output_dim))
    
    def get_A_hat(self, graph):
        A = torch.zeros((len(graph), len(graph)))
        D = torch.zeros((len(graph), len(graph)))
        for i in graph:
            A[i, i] = 1
            D[i, i] = 1
            for neighbor_tup in graph[i]:
                A[i, neighbor_tup[0]] += 1#neighbor_tup[1]
                D[i, i] += 1
            D[i, i] = 1 / torch.sqrt(D[i, i])
        return D @ A @ D
    
    def forward(self, x):
        ## Convert One-Hot To Embeddings
        x = self.layer_input_emb(x)
        #x = self.layer_input_emb2(x)
        
        ## Get Neighboring Info
#        x = self.layer_input_gcn(self.A_hat @ x)
#        x = F.relu(x)
#        x = self.layer_1_gcn(self.A_hat @ x)
#        x = F.relu(x)
#        x = self.layer_2_gcn(self.A_hat @ x)
#        x = F.relu(x)
#        x = self.layer_output_gcn(self.A_hat @ x)
        for i in range(len(self.layer_lst) - 1):
            x = self.layer_lst[i](self.A_hat @ x)
            x = F.relu(x)
        x = self.layer_lst[-1](self.A_hat @ x)
        return F.softmax(x)

def get_next_point(graph, curr, max_hop = 3):
    near_points = set([curr])
    q = [curr]
    for _ in range(max_hop):
        tmp = []
        for point in q:
            for neighbor_tup in graph[point]:
                if neighbor_tup[0] not in near_points:
                    near_points.add(neighbor_tup[0])
                    tmp.append(neighbor_tup[0])
        q = tmp
    next_point = curr
    while next_point in near_points:
        next_point = np.random.choice(len(graph))
    return next_point

def get_k_farthest_points(graph, k, max_hop=3):
    points = [np.random.choice(len(graph))]
    for _ in range(k - 1):
        points.append(get_next_point(graph, points[-1], max_hop))
    return points

def initialize_prob_matrix(graph, model):
    if model == "SIS":
        dim = 2
    else:
        dim = 3
    return torch.ones((len(graph), dim)) / dim

def vec_2_one_hot(vec, model):
    if model == "SIS":
        ret = torch.zeros((len(vec), 2))
    else:
        ret = torch.zeros((len(vec), 3))
    for i in range(len(vec)):
        ret[i, int(vec[i])] = 1
    return ret

def sample_X(vec, model, batch_size, samples = None):
    ret = initialize_prob_matrix(vec, model)
    if samples is None:
        samples = np.random.choice(len(vec), batch_size, replace=False)
    for i in samples:
        ret[i, :] = 0
        ret[i, int(vec[i])] = 1
    return ret

def to_vec(matrix):
    ret = torch.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        ret[i] = torch.argmax(matrix[i,:])
    return ret

def get_loss(output, target, metric = "CrossEntropy"):
    assert metric in ["CrossEntropy", "MSE", "Mis-rate"]
    if metric == "CrossEntropy":
        return -torch.sum(target * torch.log(output)) / target.shape[0]
    elif metric == "MSE":
        return torch.sum(torch.square(output - target)) / target.shape[0]
    else:
        output_vec = to_vec(output)
        target_vec = to_vec(target)
        return 100.0 * torch.sum((output_vec - target_vec) != 0) / len(target_vec)

def train_gcn(graph, model, X_lst, Y_lst, epoch=100, lr=1e-3, metric="CrossEntropy", train_freq=1, step_size=500, decay=1):
    bayesian_gcn = GCN(graph, model)
    if train_on_gpu:
        #X = X.to(device="cuda")
        bayesian_gcn = bayesian_gcn.cuda()
    optimizer = optim.SGD(bayesian_gcn.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay)
    
    loss_arr = []
    loss = 0
    for i in tqdm(range(epoch)):
        optimizer.zero_grad()
        idx = np.random.choice(len(X_lst))
        X = X_lst[idx]
        Y = Y_lst[idx]
        #X = vec_2_one_hot(X, model)
        if train_on_gpu:
            X = X.to(device="cuda")
            Y = Y.to(device="cuda")
        output = bayesian_gcn(X)
        loss += get_loss(output, Y, metric)
        if (i + 1) % train_freq == 0:
            loss_arr.append(float(loss.data / train_freq))
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss = 0
    
    return bayesian_gcn, loss_arr

def evaluation(bayesian_gcn, model, X_lst, Y_lst):
    cross_entropy_arr = []
    mse_arr = []
    mis_rate_arr = []
    for i in range(len(X_lst)):
        X = X_lst[i] #vec_2_one_hot(X_lst[i], model)
        Y = Y_lst[i]
        if train_on_gpu:
            X = X.to(device="cuda")
            Y = Y.to(device="cuda")
        output = bayesian_gcn(X)
        cross_entropy = float(get_loss(output, Y, "CrossEntropy").data)
        mse = float(get_loss(output, Y, "MSE").data)
        mis_rate = float(get_loss(output, Y, "Mis-rate").data)
        
        cross_entropy_arr.append(cross_entropy)
        mse_arr.append(mse)
        mis_rate_arr.append(mis_rate)
    df = pd.DataFrame.from_dict({"CrossEntropy": cross_entropy_arr, "MSE": mse_arr, "Miss-Rate": mis_rate_arr})
    return df

def visualize_prediction(graph, model, bayesian_gcn, X, Y):
    if train_on_gpu:
        X = X.to(device="cuda")
    output = bayesian_gcn(X)
    pred = to_vec(output)
    
    G_orig = nx.Graph()
    G_pred = nx.Graph()
    subset_color = ["blue", "red", "green"]
    miss = 0
    for node in tqdm(graph):
        G_orig.add_nodes_from([(node, {"color": subset_color[Y[node].argmax()]})])
        G_pred.add_nodes_from([(node, {"color": subset_color[int(pred[node].data)]})])
        if Y[node].argmax() != int(pred[node].data):
            miss += 1
    for node in tqdm(graph):
        for neighbor_tup in graph[node]:
            G_orig.add_edge(node, neighbor_tup[0], color="y")
            G_pred.add_edge(node, neighbor_tup[0], color="y")
    
    return G_orig, G_pred, miss / len(Y)

def workflow(n_nodes=100, avg_degree=3, seed=0, recover_rate=0.2, T=10, model="SIS", N_paths=1000, k=5, train_metric="CrossEntropy", epoch=1000, lr=1e-3, train_test_split=0.8, batch_size=20, n_repeat=5, max_hop=3, train_freq=1, step_size=500, decay=1, visualize_obs=0):
    graph = load_graph(n_nodes, avg_degree, seed = seed)
    sim_paths = simulate_multiple(graph, recover_rate, T, model, N_paths)
    Y_train = sim_paths[:int(N_paths * train_test_split),:]
    Y_train_lst = []
    X_train_lst = []
    Y_test_lst = []
    X_test_lst = []
    print("Converting to one-hot...")
    for i in tqdm(range(Y_train.shape[0])):
        for _ in range(n_repeat):
            Y_train_lst.append(vec_2_one_hot(Y_train[i,:], model))
            X_train_lst.append(sample_X(Y_train[i,:], model, batch_size))
    Y_test = sim_paths[int(N_paths * train_test_split):,:]
    farthest_points = get_k_farthest_points(graph, k, max_hop)
    for i in tqdm(range(Y_test.shape[0])):
        Y_test_lst.append(vec_2_one_hot(Y_train[i,:], model))
        X_test_lst.append(sample_X(Y_test[i,:], model, batch_size, farthest_points))
    print("Training...")
    bayesian_gcn, loss_arr = train_gcn(graph, model, X_train_lst, Y_train_lst, epoch, lr, train_metric, train_freq, step_size, decay)
    print("Evaluating...")
    df_results = evaluation(bayesian_gcn, model, X_test_lst, Y_test_lst)
    
    G_orig, G_pred, miss = visualize_prediction(graph, model, bayesian_gcn, X_test_lst[visualize_obs], Y_test_lst[visualize_obs])
    path_sim = GRAPH_DIR + "/PandemicSimulations/"
    title = model + "_gamma=" + str(recover_rate) + "_T=" + str(T) + "_N=" + str(N_paths) + "_k=" + str(k) + "_Obs=" + str(visualize_obs)
    fname = path_sim + title + ".png"
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    pos = nx.spring_layout(G_orig)
    nx.draw(G_orig, pos, node_size=5, edge_color=nx.get_edge_attributes(G_orig, "color").values(), node_color=nx.get_node_attributes(G_orig, "color").values())
    ax.set_title("Original")
    ax = fig.add_subplot(122)
    nx.draw(G_pred, pos, node_size=5, edge_color=nx.get_edge_attributes(G_pred, "color").values(), node_color=nx.get_node_attributes(G_pred, "color").values())
    ax.set_title("Predicted")
    plt.suptitle(title.replace("_", " ") + "\nMiss-Classification = " + str(miss * 100) + "%")
    plt.savefig(fname)
    plt.clf()
    plt.close()
    return loss_arr, df_results

loss_arr, df_results = workflow(n_nodes=1000, avg_degree=3, seed=0, recover_rate=0.2, T=10, model="SIS", N_paths=1000, k=10, train_metric="MSE", epoch=10000, lr=1e-2, train_test_split=0.8, batch_size=100, n_repeat=20, max_hop=3, train_freq=10, step_size=5000, decay=0.1, visualize_obs=0)
print(df_results)
plt.plot(loss_arr)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Cross-Entropy Loss")
plt.show()
