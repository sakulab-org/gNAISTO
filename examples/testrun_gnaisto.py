import numpy as np

from gnaisto import estimate_regulation, prepare_data, signed_edge_symmetrize
from gnaisto.data_utils import simulate_data, list_random_extract_network_matrix
from gnaisto.plot_utils import plot_all_estimated_regulation_weights

def main():
    use_example_data = True
    if use_example_data:
        fn_expression = 'example_data/expression.csv'
        fn_regulation = 'example_data/regulation.csv'
        expression, genename, samplename, regulation_test = prepare_data(fn_expression, fn_regulation)
        regulation_test = signed_edge_symmetrize(regulation_test)
        num_node_given = np.sum((regulation_test != 0).any(axis=0))
    else:
        ### Make simulated data
        num_node = 100
        num_edge = 200
        num_sample = 200
        num_node_given = 10

        expression, regulation, details = simulate_data(f'GaussScaleFreeN{num_node}E{num_edge}', num_sample)
        givennode = list_random_extract_network_matrix(regulation, [0], num_nodes=num_node_given)
        givennode = givennode[0]
        new_order = givennode + [i for i in list(range(num_node)) if i not in givennode]
        expression = expression[:, new_order]
        regulation = regulation[np.ix_(new_order, new_order)]
        regulation_test = regulation.copy()
        regulation_test[num_node_given:, :] = 0
        regulation_test[:, num_node_given:] = 0
        genename = [f"K{i}" for i in range(num_node_given)] + [f"U{i}" for i in range(num_node_given, num_node)]
        
    hypparam = {'alpha': 2,  'beta': 1, 'gamma': 1e-3}
    regulation_estim, table_regulation_estim = estimate_regulation(expression, method="gNAISTO", reg_known=regulation_test, hypparam=hypparam, num_core=8, genename=genename)
    
    table_regulation_estim.to_csv("estimated_regulation.csv", index=False)    
    if not use_example_data:
        plot_all_estimated_regulation_weights(regulation_estim[:num_node_given, num_node_given:], given=regulation[:num_node_given, num_node_given:])
        
if __name__ == "__main__":
    main()