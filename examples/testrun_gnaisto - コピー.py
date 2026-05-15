# examples/run_gnaisto_small.py

import numpy as np

from gnaisto import estimate_regulation, prepare_data
from gnaisto.data_utils import simulate_data, list_random_extract_network_matrix
from gnaisto.plot_utils import plot_all_estimated_regulation_weights

from sklearn.covariance import GraphicalLasso
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt

def scatter_regulation_estim(regulation_estim_gnaisto, regulation_estim_glasso, regulation_true):
    x = regulation_estim_gnaisto.flatten()
    y = regulation_estim_glasso.flatten()
    true = regulation_true.flatten()
    plt.figure(figsize=(6, 6))
    plt.scatter(x[true == 0], y[true == 0], color='lightgray', label='No Regulation', alpha=0.5)
    plt.scatter(x[true == 1], y[true == 1], color='blue', label='Positive Regulation', alpha=0.5)
    plt.scatter(x[true == -1], y[true == -1], color='red', label='Negative Regulation', alpha=0.5)
    plt.legend()
    plt.xlabel("gNAISTO Estimated Regulation Weights")
    plt.ylabel("Graphical Lasso Estimated Partial Correlations")
    plt.title("Comparison of Estimated Regulation Weights")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.5)   
    plt.tight_layout()
    plt.show()

def main():
    num_node = 100
    num_edge = 200
    num_node_given = 10
    flag_dataload = False
    if flag_dataload:
        fn_expression = 'example_data/expression.csv'
        fn_regulation = 'example_data/regulation.csv'
        expression, genename, samplename, regulation = prepare_data(fn_expression, fn_regulation)
    else:
        expression, regulation, details = simulate_data(f'GaussScaleFreeN{num_node}E{num_edge}', 1000)
        givennode = list_random_extract_network_matrix(regulation, [0], num_nodes=num_node_given)
        givennode = givennode[0]
        new_order = givennode + [i for i in list(range(num_node)) if i not in givennode]
        expression = expression[:, new_order]
        regulation = regulation[np.ix_(new_order, new_order)]
    regulation_forCompare = regulation.copy()
    regulation_forCompare = regulation_forCompare[num_node_given:, :num_node_given]

    regulation_test = regulation.copy()
    regulation_test[num_node_given:, :] = 0
    regulation_test[:, num_node_given:] = 0

    hypparam = {'alpha': 2,  'beta': 1, 'gamma': 1e-4}
    regulation_estim_gnaisto = estimate_regulation(expression, method="gNAISTO", reg_known=regulation_test, hypparam=hypparam, num_core=8)
    regulation_estim_gnaisto = regulation_estim_gnaisto[num_node_given:, :num_node_given]
    plot_all_estimated_regulation_weights(regulation_estim_gnaisto, given=regulation_forCompare)

    model = GraphicalLasso(alpha=hypparam["gamma"]/2) # /2 for adjusting to gLASSO and gNAISTO definitions
    model.fit(expression)
    theta = model.precision_ 
    partialcorr = -theta / np.sqrt(np.outer(np.diag(theta), np.diag(theta)))
    np.fill_diagonal(partialcorr, 0)
    regulation_estim_glasso = partialcorr.copy()
    regulation_estim_glasso = regulation_estim_glasso[num_node_given:, :num_node_given]

    AUPRC_gnaisto = average_precision_score(np.abs(regulation_forCompare.flatten()), np.abs(regulation_estim_gnaisto.flatten()))
    AUROC_gnaisto = roc_auc_score(np.abs(regulation_forCompare.flatten()), np.abs(regulation_estim_gnaisto.flatten()))
    AUPRC_glasso = average_precision_score(np.abs(regulation_forCompare.flatten()), np.abs(regulation_estim_glasso.flatten()))
    AUROC_glasso = roc_auc_score(np.abs(regulation_forCompare.flatten()), np.abs(regulation_estim_glasso.flatten()))
    print(f"gNAISTO: AUPRC={AUPRC_gnaisto:.4f}, AUROC={AUROC_gnaisto:.4f}")
    print(f"Graphical Lasso: AUPRC={AUPRC_glasso:.4f}, AUROC={AUROC_glasso:.4f}")

    scatter_regulation_estim(regulation_estim_gnaisto, regulation_estim_glasso, regulation_forCompare)

if __name__ == "__main__":
    main()