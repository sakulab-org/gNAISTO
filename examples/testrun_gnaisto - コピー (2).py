import numpy as np
import pandas as pd

from gnaisto import estimate_regulation, prepare_data, signed_edge_symmetrize
from gnaisto.data_utils import simulate_data, list_random_extract_network_matrix

def main():
    flag_dataload = True
    if flag_dataload:
        fn_expression = 'example_data/expression.csv'
        fn_regulation = 'example_data/regulation.csv'
        expression, genename, samplename, regulation_test = prepare_data(fn_expression, fn_regulation)
        regulation_test = signed_edge_symmetrize(regulation_test)
        num_node_given = np.sum((regulation_test != 0).any(axis=0))
    else:
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
        samplename = [f"Sample{i}" for i in range(num_sample)]
        
        expression_save = expression.copy()
        rng = np.random.default_rng(seed=0)
        scale = 10 ** rng.uniform(0, 3, size=num_node)
        shift = rng.uniform(3, 6, size=num_node)
        expression_save = (expression_save + shift) * scale
        expression_save = np.round(expression_save, 3)
        table_expression = pd.DataFrame(expression_save, index=samplename, columns=genename)
        table_expression.to_csv("expression.csv")
        table_regulation_test = pd.DataFrame(regulation_test, index=genename, columns=genename).stack().reset_index().rename(columns={'level_0': 'Col', 'level_1': 'Row', 0: 'Sign'})
        table_regulation_test = table_regulation_test[table_regulation_test['Sign'] != 0].reset_index(drop=True)
        table_regulation_test.to_csv("regulation.csv", index=False)

    hypparam = {'alpha': 2,  'beta': 1, 'gamma': 1e-5}
    regulation_estim, table_regulation_estim = estimate_regulation(expression, method="gNAISTO", reg_known=regulation_test, hypparam=hypparam, num_core=8, genename=genename)
    table_regulation_estim.to_csv("estimated_regulation.csv", index=False)

if __name__ == "__main__":
    main()