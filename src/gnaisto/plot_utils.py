import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
# from graphviz import Digraph, Graph

def plot_regulatory_matrix_heatmap(
        weights_T, fn=None, 
        given_T=None, given_T_T=None, blind_T=None, blind_T_T=None, annot=False, 
        title="Regulatory Weights Heatmap", 
        clim=2
        ):
    """Visualize the estimated regulatory weights as a heatmap."""
    plt.figure(figsize=np.array(weights_T.T.shape)/3 + np.array([0,4]))
    weights_T_masked = np.ma.masked_where(weights_T == 0, weights_T)
    cmap = plt.cm.get_cmap("RdYlGn").copy()
    cmap.set_bad(color='white')  
    im = plt.imshow(weights_T_masked, cmap=cmap, vmin=-clim, vmax=clim)
    plt.colorbar(im)
    for i in range(np.shape(weights_T)[0]):
        for j in range(np.shape(weights_T)[1]):
            txt = ""
            if given_T is not None and given_T[i, j] != 0:
                color = "black"
                if given_T[i, j] > 0:
                    txt = "+"
                elif given_T[i, j] < 0:
                    txt = "−"
            elif given_T_T is not None and given_T_T[i, j] != 0:
                color = "lightgray"
                if given_T_T[i, j] > 0:
                    txt = "+"
                elif given_T_T[i, j] < 0:
                    txt = "−"
            elif blind_T is not None and blind_T[i, j] != 0:
                color = "red"
                if blind_T[i, j] > 0:
                    txt = "+"
                elif blind_T[i, j] < 0:
                    txt = "−"
            elif blind_T_T is not None and blind_T_T[i, j] != 0:
                color = "blue"
                if blind_T_T[i, j] > 0:
                    txt = "+"
                elif blind_T_T[i, j] < 0:
                    txt = "−"
            if txt != "":
                # plt.text(j + 0.05, i + 0.25, txt, ha="center", va="center", color=color, fontsize=12)
                # plt.text(j+0.5, i+0.5, txt, ha="center", va="center", color=color, fontsize=12)
                plt.text(j, i, txt, ha="center", va="center", color=color, fontsize=12)
    plt.title(title)
    plt.tight_layout()
    plt.xlabel("Source Cluster")
    plt.ylabel("Target Cluster")
    if fn is not None:
        plt.savefig(fn)
        plt.close()
    else:
        plt.show()

def plot_all_estimated_regulation_weights(weights, given=None, given_T=None, fn=None):
    """Visualize the all estimated regulatory weights as bar plots.""" 
    plt.figure(figsize=(10, 8))
    if weights.ndim == 3:
        weights = np.mean(weights, axis=0)
    weights = weights.flatten()
    sortidx = np.argsort(np.abs(weights))[::-1]
    weights = weights[sortidx]

    if given is not None:
        given = given.flatten()
        given = given[sortidx]
    if given_T is not None:
        given_T = given_T.flatten()
        given_T = given_T[sortidx]

    colors = np.where(weights >= 0, 'blue', 'red')
    heights = [w if w >= 0 else abs(w) for w in weights]    
    if given is not None:
        labels = np.where(given == 1, '+', np.where(given == -1, '−', ''))
    if given_T is not None:
        labels_T = np.where(given_T == 1, '+', np.where(given_T == -1, '−', ''))

    plt.bar(range(len(weights)), heights, color=colors)
    if given is not None:
        for i, label in enumerate(labels):
            if label != '':
                plt.text(i, heights[i], label, ha='center', va='bottom', fontsize=8)
    if given_T is not None:
        for i, (label, label_T) in enumerate(zip(labels, labels_T)):
            if label == '' and label_T != '':
                plt.text(i, heights[i], label_T, ha='center', va='bottom', fontsize=6, color='gray')
                
    plt.title("Estimated Regulation Weights")
    plt.tight_layout()    
    
    if fn is not None:
        plt.savefig(fn, dpi=300)        
        plt.close()
    else:
        plt.show()

def draw_directed_graph(regulation_matrix, node_names, color_dict=None,  fillcolor_dict=None, filename_prefix="graph", flag_undirect=False):
    """    Draw a directed graph from a regulation matrix and save it as an image."""
    A = np.abs(regulation_matrix)
    if flag_undirect:
        G = nx.from_numpy_array(A)
    else:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    # Node names
    mapping = {i: name for i, name in enumerate(node_names)}
    G = nx.relabel_nodes(G, mapping)

    # Plot
    color_dict = color_dict or {}
    fillcolor_dict = fillcolor_dict or {}
    draw_graphviz_graph(G, color_dict=color_dict, filename_prefix=filename_prefix, fillcolor_dict=fillcolor_dict, flag_undirect=flag_undirect)

def draw_graphviz_graph(G_nx, color_dict=None, fillcolor_dict=None, filename_prefix="graph", flag_undirect=False):
    if flag_undirect:
        G = Graph('G', filename=filename_prefix+".gv", format='png')
    else:
        G = Digraph('G', filename=filename_prefix+".gv", format='png')
    edges = list(G_nx.edges())
    for source, target in edges:
        G.edge(source, target) 
    color_dict = color_dict or {}
    fillcolor_dict = fillcolor_dict or {}
    for node in G_nx.nodes():
        color = color_dict.get(node, 'black')
        fillcolor = fillcolor_dict.get(node, 'white')
        G.node(node, style='filled', fillcolor=fillcolor, color=color, penwidth='2')

    G.render()

def draw_graphviz_graph2(regulation_matrix, node_names, color_dict=None,  fillcolor_dict=None, filename_prefix="graph", flag_undirect=False):
    if flag_undirect:
        G = Graph('G', filename=filename_prefix+".gv", format='png')
    else:
        G = Digraph('G', filename=filename_prefix+".gv", format='png')
        
    counts = Counter(node_names)
    counter = defaultdict(int)
    node_names_new = []
    for s in node_names:
        if counts[s] > 1:
            counter[s] += 1
            node_names_new.append(f"{s} ({counter[s]})")
        else:
            node_names_new.append(s)
    node_names = node_names_new
            
    activates = np.argwhere(regulation_matrix > 0)
    inhibits = np.argwhere(regulation_matrix < 0)
    activates = [(node_names[i], node_names[j]) for i, j in activates]
    inhibits = [(node_names[i], node_names[j]) for i, j in inhibits]
    for source, target in activates:
        G.edge(source, target) 
    for source, target in inhibits:
        G.edge(source, target, arrowhead='tee') 

    color_dict = color_dict or {}
    fillcolor_dict = fillcolor_dict or {}
    for node in node_names:
        color = color_dict.get(node, 'black')
        fillcolor = fillcolor_dict.get(node, 'white')
        G.node(node, style='filled', fillcolor=fillcolor, color=color, penwidth='2')

    G.render()
