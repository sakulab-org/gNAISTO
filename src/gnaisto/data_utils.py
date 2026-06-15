import itertools
import pickle
import re
import os
import pandas as pd
import numpy as np
import networkx as nx
import random
from scipy.stats import boxcox, zscore

# ===========================================
# Data Preparation
# ===========================================

def prepare_data(expression_file, regulation_file, flag_boxcox=False, flag_transpose=False, flag_normalize=True, flag_noiso=False, gt_expression=None):
    """Load gene expression and regulation data, and process clustering."""
    if expression_file.endswith('.csv'):
        sep = ','
    elif expression_file.endswith('.tsv'):
        sep = '\t'

    ## Load expression data
    # Check if the first row or column contains strings
    expr_data_check = pd.read_csv(expression_file, header=None, sep=sep, nrows=5)
    first_row = expr_data_check.iloc[0, 1:]
    first_col = expr_data_check.iloc[1:, 0]
    flag_firstrowstr = first_row.apply(lambda x: not str(x).replace('.', '', 1).isdigit()).any()
    flag_firstcolstr = first_col.apply(lambda x: not str(x).replace('.', '', 1).isdigit()).any()
    if flag_firstrowstr:
        index_header = 0
    else:
        index_header = None
    if flag_firstcolstr:
        index_col = 0
    else:
        index_col = None
    # load
    expr_data = pd.read_csv(expression_file, header=index_header, sep=sep, index_col=index_col)
    if flag_transpose:
        expr_data = expr_data.T
    genename = expr_data.columns.to_numpy()
    samplename = expr_data.index.to_numpy()
    expr_data = expr_data.to_numpy()
    
    print("  Genes (first 5):", genename[:5])

    # Filter genes by mean expression if gt_expression is specified
    if gt_expression is not None:
        print(f"  Filtering genes with mean expression <= {gt_expression}...")
        mean_expression = np.mean(expr_data, axis=0)
        idx_gt = mean_expression > gt_expression
        expr_data = expr_data[:, idx_gt]
        genename = genename[idx_gt]

    # regulation_data: row=source, col=target, value=effect
    if regulation_file.endswith('.csv'):
        sep = ','
    elif regulation_file.endswith('.tsv'):
        sep = '\t'
    regulation_data = pd.read_csv(regulation_file, header=None, sep=sep, names=["Source", "Target", "Effect"])

    if regulation_data["Effect"].isin(["+", "-"]).all():
        # Convert "+" and "-" to 1 and -1
        regulation_data["Effect"] = regulation_data["Effect"].map({"-": -1, "+": 1})
    regulation_data = regulation_data[regulation_data["Source"].isin(genename) & regulation_data["Target"].isin(genename) & (regulation_data["Effect"] != 0)]

    regulation_matrix = np.zeros((len(genename), len(genename)))
    for _, row in regulation_data.iterrows():
        source_idx = np.where(genename == row["Source"])[0]
        target_idx = np.where(genename == row["Target"])[0]
        regulation_matrix[source_idx, target_idx] = row["Effect"]
    np.fill_diagonal(regulation_matrix, 0)
    
    # Reorder genes by whether they have known regulation
    has_known_regulation = (np.sum(regulation_matrix != 0, axis=0) > 0) | (np.sum(regulation_matrix != 0, axis=1) > 0)
    new_order = np.concatenate([np.where(has_known_regulation)[0], np.where(~has_known_regulation)[0]])
    expr_data = expr_data[:, new_order]
    genename = genename[new_order]
    regulation_matrix = regulation_matrix[np.ix_(new_order, new_order)]

    # Filter out isolated genes if flag_noiso is True 
    if flag_noiso:
        print(f"  Filtering isolated genes...")
        is_isolate = (np.sum(np.abs(regulation_matrix), axis=0) == 0) & (np.sum(np.abs(regulation_matrix), axis=1) == 0)
        expr_data = expr_data[:, ~is_isolate]
        genename = genename[~is_isolate]
        regulation_matrix = regulation_matrix[~is_isolate][:, ~is_isolate]
        
    # Transform expression data by Box-Cox if flag_boxcox is True
    expr_data_transform = expr_data.copy()
    for i in range(expr_data.shape[1]):
        expr_data_now = expr_data[:,i]
        expr_data_now = expr_data_now + 1e-10
        if flag_boxcox:
            expr_data_now, _ = boxcox(expr_data_now)
        if flag_normalize:
            expr_data_now = zscore(expr_data_now)/100
        expr_data_transform[:, i] = expr_data_now
    expr_data = expr_data_transform.copy()
    if flag_boxcox:
        print("  Box-Cox transformation applied.")
    if flag_normalize:
        print("  Expression data normalized by z-score.")

    return expr_data, genename, samplename, regulation_matrix

def convert_regulation_to_clusters(regulation_data, cluster_data):
    """Convert gene-level regulation to cluster-level regulation."""
    regulation_cluster_matrix = np.zeros((cluster_data["Cluster"].nunique(), cluster_data["Cluster"].nunique()))
    for _, row in regulation_data.iterrows():
        source_cluster = cluster_data.loc[row["Source"]]["Cluster"]
        target_cluster = cluster_data.loc[row["Target"]]["Cluster"]
        regulation_cluster_matrix[source_cluster, target_cluster] += row["Effect"]
    return regulation_cluster_matrix

def simulate_data(dataname, num_data, seed=None, noisestrength=1e-2):
    """Simulate gene expression data and regulation matrix based on the dataset name."""
    np.random.seed(seed)
    random.seed(seed)
    details = {}
    
    m = re.match(r'GaussScaleFreeN(\d+)E(\d+)', dataname)
    if m:
        nodenum = int(m.group(1))
        edgenum = int(m.group(2))
        G = nx.barabasi_albert_graph(nodenum, round(edgenum/nodenum), seed=seed)
        A = nx.to_numpy_array(G)
        regulation_matrix = A.copy()
        precision_matrix = A.copy()
        # Assign random signs and weights to edges
        for i in range(nodenum):
            for j in range(i, nodenum):
                if regulation_matrix[i, j] == 1:
                    regulation_matrix[i, j] = np.random.choice([-1, 1])
                    regulation_matrix[j, i] = regulation_matrix[i, j]
                    precision_matrix[i, j] = -regulation_matrix[i, j] * np.random.uniform(0.5, 1.5)
                    precision_matrix[j, i] = precision_matrix[i, j]
                    
                    
        np.fill_diagonal(precision_matrix, 1.0)
        w = np.linalg.eigvalsh(precision_matrix)
        lam_min = w[0]
        if lam_min <= 0:
            tau = -lam_min + 1e-3
            precision_matrix += tau * np.eye(nodenum)
        # Simulate expression data
        covariance_matrix = np.linalg.inv(precision_matrix + nodenum * np.eye(nodenum) * noisestrength)
        expr_data = np.random.multivariate_normal(mean=np.zeros(nodenum), cov=covariance_matrix, size=num_data)
        details['precision_matrix'] = precision_matrix

    else:
        raise ValueError(f"Unknown dataname: {dataname}")
    return expr_data, regulation_matrix, details

# ===========================================
# Network utils
# ===========================================

def largest_connected_component_nodes(regulation_matrix):
    """Find the largest connected component in the regulation matrix."""
    A = np.abs(regulation_matrix)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    largest = max(nx.weakly_connected_components(G), key=len) 
    largest_nodes = list(largest)
    return largest_nodes

def list_random_extract_network_matrix(regulation_matrix, networkidx, num_nodes=10, ub_overlapratio=0.5, seed=None):
    """Randomly extract a subnetwork from the regulation matrix."""
    print(f"  Number of nodes to be extracted: {num_nodes}")
    print(f"  Permitted overlap ratio between subnetworks: {ub_overlapratio}")
    maxnetworkidx = max(networkidx)

    # candidates is the largest connected component of regulator network
    network = np.abs(regulation_matrix)
    candidates = np.where(np.sum(network, axis=1) > 0)[0].tolist()
    notcandidates = [i for i in range(len(network)) if i not in candidates]
    candidate_network = network.copy()
    candidate_network[notcandidates, :] = 0
    candidate_network[:, notcandidates] = 0
    
    largest_nodes = largest_connected_component_nodes(candidate_network)
    notlargest_nodes = [i for i in range(len(network)) if i not in largest_nodes]
    candidate_network[notlargest_nodes, :] = 0
    candidate_network[:, notlargest_nodes] = 0
    candidates = largest_nodes
    if len(candidates) < num_nodes:
        raise ValueError("Not enough candidate nodes to extract the desired subnetwork.")
    
    print(f"  Number of candidate nodes for extraction: {len(candidates)}")
    candidate_G = nx.from_numpy_array(candidate_network, create_using=nx.DiGraph)
    if seed is None:
        seed = random.randint(0, 1000000)
    list_givennode = []
    tryidx = 0
    for i in range(maxnetworkidx+1):
        flag_fail = True
        while flag_fail:
            flag_fail = False
            tryidx+=1
            rng = random.Random(seed + tryidx)
            givennode = sample_connected_subgraph_randomwalk(candidate_G.to_undirected(), num_nodes, rng)
            flag_fail = len(givennode) < num_nodes
            
            # overlap check
            if flag_fail == False:
                overlapratio = 0
                for j in range(i):
                    overlap = len(set(givennode).intersection(set(list_givennode[j])))
                    overlapratio = max(overlapratio, overlap / num_nodes)
                if overlapratio > ub_overlapratio:
                    flag_fail = True

            if tryidx > 10000:
                raise ValueError("list_random_extract_network_matrix failed to find suitable given nodes.")
        # print(f"    Found subnetwork {i} after {tryidx} tries.")
        list_givennode.append(sorted(givennode))
        
    print(f"  Number of tries to find given nodes: {tryidx}")
    return list_givennode

def sample_connected_subgraph_randomwalk(G, k, rng):
    start = rng.choice(list(G.nodes))
    visited = {start}
    cur = start

    while len(visited) < k:
        nbrs = list(G.neighbors(cur))
        if not nbrs:
            break
        cur = rng.choice(nbrs)
        visited.add(cur)

    return sorted(list(visited))

def load_network_matrix(dir_load, networkidx, num_nodes=10):
    """Load a subnetwork from a saved file."""
                # new_order = givennode + [i for i in list(range(num_genes)) if i not in givennode]
                # with open(os.path.join(dir_output_each, f"Run{i}_test_regulation.pkl"), 'wb') as f:
                #     pickle.dump({"test_regulation_correct": test_regulation_correct, "test_expr": test_expr, "networksize": networksize, "new_order": new_order}, f)
    list_givennode = []
    for i in networkidx:
        with open(os.path.join(dir_load, f"Run{i}_test_regulation.pkl"), 'rb') as f:
            data_loaded = pickle.load(f)
            givennode = data_loaded['new_order'][:num_nodes]
            list_givennode.append(sorted(givennode))
    return list_givennode

def sample_notexisting_edges(regulation_matrix, ratio_samples, seed=None, flag_direct=False, num_divide=1):
    """Sample non-existing edges from the regulation matrix."""
    random.seed(seed)
    networksize = regulation_matrix.shape[0]
    if flag_direct: 
        pairs_all = list(itertools.permutations(range(networksize), 2))
        row, col = np.where(regulation_matrix != 0)
    else:
        pairs_all = list(itertools.combinations(range(networksize), 2))
        row, col = np.where(np.triu(regulation_matrix, k=1) != 0)
    pairs_exist = list(zip(row, col))

    num_pairs_fp = int(len(pairs_exist) * ratio_samples)
    pairs_notexist = list(set(pairs_all) - set(pairs_exist)) 
    pairs_sample = random.sample(pairs_notexist, num_pairs_fp)
    if num_divide == 1:
        return pairs_sample
    else:
        tempprob = np.random.rand(len(pairs_sample))
        list_pairs_sample_divided = []
        for i in range(num_divide):
            pairs_fp_i = [pairs_sample[j] for j in range(len(pairs_sample)) if (tempprob[j] >= i/num_divide) and (tempprob[j] < (i+1)/num_divide)]
            list_pairs_sample_divided.append(pairs_fp_i)
        return list_pairs_sample_divided

def mark_neighbors(regulation, rootgenes):
    """Mark neighbors of given nodes in the regulation matrix."""
    A = np.abs(regulation)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    colnames = ["Regulator", "Target", "2up", "2down", "Fork"]
    np_mark = np.empty((len(G), len(colnames)), dtype=object)
    for i in range(len(G)):
        for j in range(len(colnames)):
            np_mark[i, j] = []

    for root in rootgenes:
        regulator_now = list(G.predecessors(root))
        target_now = list(G.successors(root))
        regulator_2up_now = []
        target_2down_now = []
        fork_now = []
        for reg in regulator_now:
            reg_2up = list(G.predecessors(reg))
            regulator_2up_now.extend(reg_2up)
            frk = list(G.successors(reg))
            fork_now.extend(frk)
        regulator_2up_now = list(set(regulator_2up_now))
        fork_now = list(set(fork_now)-{root})
        for tar in target_now:
            tar_2down = list(G.successors(tar))
            target_2down_now.extend(tar_2down)
        target_2down_now = list(set(target_2down_now))
        
        for i in regulator_now:
            np_mark[i, 0].append(root)
        for i in target_now:
            np_mark[i, 1].append(root)
        for i in regulator_2up_now:
            np_mark[i, 2].append(root)
        for i in target_2down_now:
            np_mark[i, 3].append(root)
        for i in fork_now:
            np_mark[i, 4].append(root)
    
    np_mark_str = np.empty(len(G), dtype=object)
    for i in range(len(G)):
        np_mark_str[i] = 'Other'
        for j in reversed(range(len(colnames))):
            if len(np_mark[i, j]) > 0:
                np_mark_str[i] = colnames[j]

    df_mark = pd.DataFrame(np_mark, index=G.nodes(), columns=colnames)
    df_mark['Mark'] = np_mark_str
    df_mark.loc[rootgenes, 'Mark'] = "Root"
    return df_mark

def signed_edge_symmetrize(A, verbose=False):
    """Make the adjacency matrix sign-symmetric."""
    A_sym = A.copy()
    A_T = A.T.copy()
    A_sym[(A == 0) & (A_T == 1)] = 1
    A_sym[(A == 0) & (A_T == -1)] = -1
    A_sym[(A == 1) & (A_T == -1)] = 0
    A_sym[(A == -1) & (A_T == 1)] = 0

    num_negativeloop = np.sum(((A == 1) & (A_T == -1)) | ((A == -1) & (A_T == 1))) / 2
    if num_negativeloop > 0 and verbose:
        print(f"signed_edge_symmetrize caution: Number of negative loops: {num_negativeloop}")

    if np.array_equal(A_sym, A) and verbose:
        print("signed_edge_symmetrize: A is already sign-symmetric.")
    return A_sym