import time
import itertools
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import tempfile
import warnings
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.covariance import GraphicalLasso
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import zscore
from collections import defaultdict
from joblib import Parallel, delayed, dump, load

from .data_utils import signed_edge_symmetrize
from .optim_utils import solve_nonsmooth_weighted_regression, objective_graphgglasso, objective_graphwlasso

methodlist_all = ["NAISTO", "g2LASSO", "gNAISTO", "g3LASSO", "gNAISTOz", "pergenewLASSO", "pergenewgLASSO", "wgLASSO", "gLASSO", "S3VM", "GENIE3", "PORTIA", "PORTIAnodir", "KOZscore"]
methodlist_unsupervised = ["GENIE3", "gLASSO"]
methodlist_semisupervised = [m for m in methodlist_all if not m in methodlist_unsupervised]
methodlist_directed = ["NAISTO", "g2LASSO", "pergenewLASSO", "GENIE3", "PORTIA", "KOZscore"]
methodlist_undirected = [m for m in methodlist_all if not m in methodlist_directed]

def make_printer(flag_parallel=False):
    msg_list = []
    def print_(text="", end="\n"):
        if flag_parallel:
            msg_list.append(str(text) + end)
        else:
            print(text, end=end)
    return print_, msg_list

# ================================
# Estimations 
# ================================
def estimate_regulation(expr, method, hypparam, **kwargs):
    """Estimate regulatory interactions with prior knowledge."""
    print("")
    # centering expr
    if not method in ["S3VM"]:
        print("Centering expression data...")
        mean_expr = expr.mean(axis=0, keepdims=True)
        expr = expr - mean_expr

    num_core = kwargs.get('num_core', 1)
    reg_known = kwargs.get('reg_known', None)
    if method in methodlist_undirected and method in methodlist_semisupervised:
        print("Making known regulation undirected...")
        reg_known = signed_edge_symmetrize(reg_known)
    if method in methodlist_semisupervised and np.all(reg_known == 0):
        raise ValueError(f"estimate_regulation - {method}: No known regulation provided after making undirected.")
    
    num_gene_reg_known = kwargs.get('num_gene_reg_known', None)
    if reg_known is not None and num_gene_reg_known is None:
        is_reg_given = (np.abs(reg_known).sum(axis=1) > 0) | (np.abs(reg_known).sum(axis=0) > 0)
        maxidx_reg_given = np.max(np.nonzero(is_reg_given))
        num_gene_reg_known = maxidx_reg_given+1

    genename = kwargs.get('genename', None)
    if genename is None:
        genename = [f"Gene{i}" for i in range(expr.shape[1])]

    print(f"Estimating regulation with method: {method}...")
    if method in  ["NAISTO", "g2LASSO", "gNAISTO", "g3LASSO", "gNAISTOz", "pergenewLASSO", "pergenewgLASSO", "wgLASSO"]:
        reg_prior = kwargs.get('reg_prior', None)
        opt_gregress = kwargs.get('opt_gregress', {})
        reg_estim = estimate_regulation_naisto(expr, reg_known, num_gene_reg_known, method, hypparam, reg_prior=reg_prior, num_core=num_core, opt_gregress=opt_gregress)
    if method in ["gLASSO"]:
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model = GraphicalLasso(alpha=hypparam["gamma"]/2) # /2 for adjusting to wgLASSO and gNAISTO definitions
        model.fit(expr)
        warnings.filterwarnings("default", category=ConvergenceWarning)
        theta = model.precision_ 
        partialcorr = -theta / np.sqrt(np.outer(np.diag(theta), np.diag(theta)))
        np.fill_diagonal(partialcorr, 0)
        reg_estim = partialcorr.copy()
    if method in  ["S3VM"]:
        inputdesign = kwargs.get('inputdesign', "concat")
        savememory = kwargs.get('savememory', False)
        flag_1svm = kwargs.get('flag_1svm', False)
        seed_s3vm_falsesammpling = kwargs.get('seed_s3vm_falsesammpling', None)
        reg_estim = estimate_regulation_s3vm(expr, reg_known, hypparam, inputdesign=inputdesign, savememory=savememory, num_gene_reg_known=num_gene_reg_known, flag_1svm=flag_1svm, seed_s3vm_falsesammpling=seed_s3vm_falsesammpling)
    if method in  ["GENIE3"]:
        idx_regulator = kwargs.get('idx_regulator', None)
        num_gene_target = num_gene_reg_known
        reg_estim = estimate_regulation_genie3(expr, num_gene_target=num_gene_target, num_core=num_core, idx_regulator=idx_regulator)
    if method == 'PORTIA' or method == 'PORTIAnodir':
        # PORTIA method
        reg_estim = estimate_regulation_portia(expr, method, hypparam, sampleidx_KO=kwargs.get('sampleidx_KO', None))
    if method == 'KOZscore':
        # KOZscore method
        sampleidx_KO = kwargs.get('sampleidx_KO')
        print("Computing Z-scores in KO data...")
        num_genes = expr.shape[0]
        Z = zscore(expr, axis=1)
        zscore_KO = np.zeros([num_genes, num_genes], dtype=float)
        for i, sampleidx_KO_now in enumerate(sampleidx_KO):
            zscore_KO[i, :] = Z[:, sampleidx_KO_now]
        reg_estim = zscore_KO

    table_reg_estim = pd.DataFrame(reg_estim[:num_gene_reg_known], index=genename[:num_gene_reg_known], columns=genename).stack().reset_index().rename(columns={'level_0': 'Col', 'level_1': 'Row', 0: 'Weight'})
    table_reg_estim['absWeight'] = np.abs(table_reg_estim['Weight'])
    table_reg_estim = table_reg_estim.sort_values('absWeight', ascending=False)
    table_reg_estim = table_reg_estim.loc[:, ['Col', 'Row', 'Weight']].reset_index(drop=True)

    return reg_estim, table_reg_estim

def estimate_regulation_naisto(expr, reg_known, num_gene_reg_known, method, hypparam, reg_prior=None, num_core=1, opt_gregress={}):
    methodlist_all = ["NAISTO", "g2LASSO", "gNAISTO", "g3LASSO", "gNAISTOz", "pergenewLASSO", "pergenewgLASSO", "wgLASSO"]
    if method in ["NAISTO", "g2LASSO", "gNAISTO", "g3LASSO", "gNAISTOz"]:
        method_regress = "GGlasso" 
        alpha = hypparam.get('alpha', None)
        beta = hypparam.get('beta', None)
        gammamode = hypparam.get('gammamode', None)
        if alpha is None or beta is None:
            if gammamode is not None and beta is not None:
                alpha = gammamode * beta + 1
                hypparam['alpha'] = alpha
    elif method in ["pergenewLASSO", "pergenewgLASSO", "wgLASSO"]:
        method_regress = "wLasso"
    if method in ["gNAISTO", "g3LASSO", "gNAISTOz"]:
        method_glasso = "gammaguided"
    elif method in ["pergenewgLASSO", "wgLASSO"]:
        method_glasso = "weighted"

    reg_given = reg_known[:num_gene_reg_known,:][:,:num_gene_reg_known]
    expr_reggiven = expr[:,:num_gene_reg_known].copy()
    num_gene = expr.shape[1]

    num_optim = 0
    num_iter = 0
    reg_estim = np.zeros([expr.shape[1], expr.shape[1]], dtype=float)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if method in ["gNAISTO", "pergenewgLASSO", "g3LASSO", "wgLASSO", "gNAISTOz"]:
        # gNAISTO/pergenewgLASSO - evaluate genes one by one, graphically
        flag_Wupdate = False
        flag_diagregularize = False
        
        if method in ["gNAISTO", "pergenewgLASSO", "gNAISTOz"]:
            print("Construct graphical regression model for known genes and one additional gene...")
            flag_zerofix = method in ["gNAISTOz"]
            list_reg_allestim = []
            flag_parallel = num_core>1
            list_theta_init = None
            if not flag_parallel:
                for geneidx_add in range(num_gene_reg_known, num_gene):
                    print()
                    print(f"Evaluating regulatory effects of gene {geneidx_add}...")
                    # Extend gene means with the new gene
                    _, theta_now, reg_allestim_now, num_optim_each, num_iter_each, msg_list = estimate_precision_by_nonsmooth_weighted_regression_single_add_gene(
                        expr_reggiven, reg_given, expr, num_gene_reg_known, geneidx_add, method_glasso, hypparam, 
                        flag_Wupdate=flag_Wupdate, flag_diagregularize=flag_diagregularize, flag_parallel=flag_parallel, flag_zerofix=flag_zerofix, list_theta_init=list_theta_init, **opt_gregress
                    )

                    reg_estim[geneidx_add, :num_gene_reg_known+1] = reg_allestim_now[-1, :]
                    reg_estim[:num_gene_reg_known+1, geneidx_add] = reg_allestim_now[-1, :]
                    reg_estim[geneidx_add, geneidx_add] = 0
                    num_optim += num_optim_each
                    num_iter += num_iter_each
                    list_reg_allestim.append(reg_allestim_now)
                    if list_theta_init is None:
                        list_theta_init = []
                    list_theta_init.append(theta_now)
                    
            else:
                tmpdir = tempfile.mkdtemp()
                expr_path = os.path.join(tmpdir, "expr_memmap")
                dump(expr, expr_path)
                expr_shared = load(expr_path, mmap_mode='r')
                
                step = num_core
                start = num_gene_reg_known
                end = min(num_gene, start + step)
                while start < num_gene:
                    # Calculate
                    results = Parallel(n_jobs=num_core, verbose=10)(
                        delayed(estimate_precision_by_nonsmooth_weighted_regression_single_add_gene)(
                            expr_reggiven, reg_given, expr_shared, num_gene_reg_known, geneidx_add, method_glasso, hypparam, 
                            flag_parallel=flag_parallel, flag_Wupdate=flag_Wupdate, flag_diagregularize=flag_diagregularize, flag_zerofix=flag_zerofix, list_theta_init=list_theta_init, **opt_gregress
                        ) for geneidx_add in range(start, end)
                    )
                    # Save
                    for geneidx_add, theta_now, reg_allestim_now, num_optim_each, num_iter_each, msg_list in results:
                        print()
                        print(f"Evaluating regulatory effects of gene {geneidx_add}...")
                        
                        reg_estim[geneidx_add, :num_gene_reg_known+1] = reg_allestim_now[-1, :]
                        reg_estim[:num_gene_reg_known+1, geneidx_add] = reg_allestim_now[-1, :]
                        reg_estim[geneidx_add, geneidx_add] = 0
                        num_optim += num_optim_each
                        num_iter += num_iter_each
                        list_reg_allestim.append(reg_allestim_now) # Parallel guarantees order
                        if list_theta_init is None:
                            list_theta_init = []
                        list_theta_init.append(theta_now)
                        for msg in msg_list:
                            print(msg, end="")
                    # Update for next batch
                    step = step*2
                    start = end
                    end = min(num_gene, start + step)

        elif method in ["g3LASSO", "wgLASSO"]:
            print("Construct graphical regression model for known genes and all unknown genes...")
            isReg_zero = None
            theta, num_loop, obj_optim, num_optim, num_iter, msg_list = estimate_precision_by_nonsmooth_weighted_regression(
                expr, -reg_known, method_glasso, hypparam, flag_Wupdate=flag_Wupdate, flag_diagregularize=flag_diagregularize, isZ=isReg_zero, **opt_gregress
            )
            partialcorr = -theta / np.sqrt(np.outer(np.diag(theta), np.diag(theta)))
            np.fill_diagonal(partialcorr, 0)
            reg_allestim = partialcorr.copy()
            reg_estim = partialcorr.copy()
    elif method == "NAISTO" or method == "pergenewLASSO":
        # NAISTO/pergenewLASSO - evaluate genes one by one
        print("Construct regression model for known genes and one additional gene...")
        list_reg_allestim = []
        for geneidx_add in range(num_gene_reg_known, num_gene):
            print(f"Evaluating regulatory effects of gene {geneidx_add}...")
            # Extend gene means with the new gene
            expr_now = np.hstack([expr_reggiven, expr[:,[geneidx_add]]])
            list_reg_allestim_now = []
            for geneidx_y in range(num_gene_reg_known):
                print(f"Target gene {geneidx_y}...")
                y = expr_now[:, geneidx_y]
                X = expr_now.copy()
                X = np.delete(X, geneidx_y, axis=1)
                P_Q = reg_given[:, geneidx_y].copy()
                P_Q = np.delete(P_Q, geneidx_y, axis=0)
                P_Q = np.append(P_Q, 0.0)
                np.seterr(divide='ignore')
                reg_allestim_now, is_failed, obj_optim, num_optim_each, num_iter_each = solve_nonsmooth_weighted_regression(
                    X, y, P_Q, method_regress, hypparam
                )
                np.seterr(divide='warn')

                reg_estim[geneidx_add, geneidx_y] = reg_allestim_now[-1]
                num_optim += num_optim_each
                num_iter += num_iter_each
                list_reg_allestim_now.append(reg_allestim_now)
            list_reg_allestim.append(np.stack(list_reg_allestim_now, axis=1))
        median_reg_allestim = np.median(np.stack(list_reg_allestim, axis=2), axis=2)
        reg_estim[:num_gene_reg_known, :num_gene_reg_known] = median_reg_allestim

    elif method == "g2LASSO":
        # g2LASSO - evaluate all genes at once
        print("Construct regression model for all genes...")
        print("Evaluating regulatory effects of all genes at once...")
        for geneidx_y in range(num_gene_reg_known):
            print(f"  Target gene {geneidx_y}...")
            y = expr[:, geneidx_y].copy()
            X = expr.copy()
            X = np.delete(X, geneidx_y, axis=1)
            P_Q = reg_known[:, geneidx_y].copy()
            P_Q = np.delete(P_Q, geneidx_y, axis=0)
            np.seterr(divide='ignore')
            reg_allestim_now, is_failed, obj_optim, num_optim_each, num_iter_each = solve_nonsmooth_weighted_regression(
                X, y, P_Q, hypparam
            )
            np.seterr(divide='warn')
            
            rowidx_now = np.arange(num_gene)
            rowidx_now = rowidx_now[rowidx_now != geneidx_y]
            reg_estim[rowidx_now, geneidx_y] = reg_allestim_now
            num_optim += num_optim_each
            num_iter += num_iter_each
    
    print(f"Estimation completed with {num_optim} optimizations and {num_iter} iterations.")

    return reg_estim

def estimate_regulation_portia(expr, method, hypparam, sampleidx_KO=None):
    expr = expr.T
    alpha1 = hypparam.get('alpha1')
    S = np.cov(expr)
    S_bar = alpha1 * np.eye(S.shape[0]) + (1 - alpha1) * S
    Theta = np.linalg.inv(S_bar)
    M1 = np.abs(Theta)
    np.fill_diagonal(M1, 0)

    Fi = np.mean(M1, axis=1)
    Fj = np.mean(M1, axis=0)
    rcw = Fi[:, np.newaxis] * Fj[np.newaxis, :]
    M2 = M1 / (2. * (rcw + 1e-50))
    np.fill_diagonal(M2, 0)
    
    def all_linear_regressions(X, _lambda):
        n_genes = X.shape[1]
        beta = []
        for j in range(n_genes):
            mask = np.ones(n_genes, dtype=bool)
            mask[j] = 0
            X_j = X[:, mask]
            y_j = X[:, j]
            cov = _lambda * np.eye(n_genes - 1) + (1. - _lambda) * np.cov(X_j.T)
            w = np.zeros(n_genes)
            w[mask] = np.linalg.inv(cov) @ X_j.T @ y_j
            assert not np.any(np.isnan(w))
            beta.append(w)
        beta = np.asarray(beta).T
        np.fill_diagonal(beta, 0)
        return beta
    
    alpha2 = hypparam.get('alpha2')
    beta = all_linear_regressions(expr.T, _lambda=alpha2)
    beta = np.abs(beta)
    np.fill_diagonal(beta, 0)
    mask = beta == 0
    beta2 = beta.copy()
    beta2[mask] = 1
    beta2 = beta2 / np.maximum(beta2, beta2.T) # almost 1
    beta2[mask] = 0
    M3 = M2 * beta2 
    if sampleidx_KO is None:
        M4 = M3.copy() 
    else:
        Z_ko, _ = estimate_regulation(expr, 'KOZscore', hypparam=None, sampleidx_KO=sampleidx_KO)
        M4 = M3 * Z_ko
    std_M4 = np.std(M4, axis=1) # Prioritization to genes with high variance in coefficients -> Hub genes. Why not mean(abs())?
    M5 = M4 * std_M4[:, np.newaxis]
    std_M3 = np.std(M3, axis=1)
    M5_fromM3 = M3 * std_M3[:, np.newaxis]
    
    if method == 'PORTIAnodir':
        reg_estim = M5_fromM3.copy()
    elif method == 'PORTIA':
        reg_estim = M5.copy()
    reg_estim = reg_estim / np.median(np.abs(reg_estim))
    # Implement the PORTIA regulation estimation logic here
    return reg_estim

def estimate_regulation_s3vm(expr, reg_known_nodir, hypparam, inputdesign="concat", savememory=False, num_gene_reg_known=None, flag_1svm=False, seed_s3vm_falsesammpling=None):
    """Estimate regulatory interactions using semi supervised SVM classification."""
    from sklearn.svm import LinearSVC
    maxratio_PN = 100

    expr_T = expr.T
    num_sample = expr.shape[0]       
    num_gene = expr.shape[1]
    C = hypparam.get('C')
    if inputdesign == "hadamard":
        xcolnum = num_sample
    elif inputdesign == "kron":
        xcolnum = num_sample * num_sample
    elif inputdesign == "concat":
        xcolnum = num_sample * 2
    
    num_pairs_all = num_gene*(num_gene-1)//2
    ij_pairs_all = list(itertools.combinations(range(num_gene), 2))
    i, j = np.where(np.triu(reg_known_nodir!=0))
    ij_pairs_true = [(int(a), int(b)) for a, b in zip(i, j)]    
    
    ij_pairs = ij_pairs_all
    if savememory and len(ij_pairs_true)*(maxratio_PN+1) < num_pairs_all:
        num_pairs_false_sample = maxratio_PN * len(ij_pairs_true)
        print(f"Too many gene pairs ({num_pairs_all}) for the given memory limit. Sampling {num_pairs_false_sample} false pairs for training SVM...")
        
        ij_pairs_false_all = list(itertools.combinations(range(num_gene), 2))
        ij_pairs_false_all = [pair for pair in ij_pairs_false_all if pair not in ij_pairs_true]

        random.seed(seed_s3vm_falsesammpling)
        if num_gene_reg_known is None:
            ij_pairs_false_regknown = []
            ij_pairs_false_sample = random.sample(ij_pairs_false_all, num_pairs_false_sample)
        else:
            ij_pairs_false_regknown = list(itertools.combinations(range(num_gene_reg_known), 2))
            ij_pairs_false_regknown = [pair for pair in ij_pairs_false_regknown if pair not in ij_pairs_true]            
            ij_pairs_false_temp = [pair for pair in ij_pairs_false_all if pair not in ij_pairs_false_regknown]
            if len(ij_pairs_false_temp) <= num_pairs_false_sample:
                ij_pairs_false_sample = ij_pairs_false_temp
            else:
                ij_pairs_false_sample = random.sample(ij_pairs_false_temp, num_pairs_false_sample)
        
        ij_pairs_false = ij_pairs_false_sample + ij_pairs_false_regknown
        ij_pairs = ij_pairs_true + ij_pairs_false
        print(f"Sampled {len(ij_pairs_true)} positive pairs and {len(ij_pairs_false)} negative pairs for training SVM.")

    num_pairs = len(ij_pairs)
    X = np.zeros((num_pairs, xcolnum), dtype=float)
    y1 = np.zeros(num_pairs, dtype=float)
    y2 = np.zeros(num_pairs, dtype=float)
    y = np.zeros(num_pairs, dtype=float)
    ii=0
    for i, j in ij_pairs:
        xi = expr_T[i]
        xj = expr_T[j]
        if inputdesign == "hadamard":
            X[ii, :] = xi * xj
        elif inputdesign == "kron":
            X[ii, :] = np.kron(xi, xj)
        elif inputdesign == "concat":
            X[ii, :num_sample] = xi
            X[ii, num_sample:] = xj
        y1[ii] = reg_known_nodir[i, j]==1
        y2[ii] = reg_known_nodir[i, j]==-1
        y[ii] = reg_known_nodir[i, j]!=0
        ii += 1
    clf1 = None
    clf2 = None
    clf = None
    if np.sum(y1)>0 and not flag_1svm:
        print(f"Training SVM for positive regulation with {np.sum(y1)} positive samples and {len(y1)-np.sum(y1)} negative samples...")
        clf1 = LinearSVC(C=C, class_weight='balanced', max_iter=10000)
        clf1.fit(X, y1)
        w1 = clf1.coef_.ravel()
        b1 = clf1.intercept_[0]        
    if np.sum(y2)>0 and not flag_1svm:
        print(f"Training SVM for negative regulation with {np.sum(y2)} positive samples and {len(y2)-np.sum(y2)} negative samples...")
        clf2 = LinearSVC(C=C, class_weight='balanced', max_iter=10000)
        clf2.fit(X, y2)
        w2 = clf2.coef_.ravel()
        b2 = clf2.intercept_[0]        
    if flag_1svm:
        print(f"Training SVM for regulation with {np.sum(y)} positive samples and {len(y)-np.sum(y)} negative samples...")
        clf = LinearSVC(C=C, class_weight='balanced', max_iter=10000)
        clf.fit(X, y)
        w = clf.coef_.ravel()
        b = clf.intercept_[0]        
    
    reg_estim = np.zeros((num_gene, num_gene), dtype=float)
    
    print("Predicting regulatory effects for all gene pairs...")
    for i in range(num_gene-1):
        X_block = np.zeros((num_gene-1-i, xcolnum), dtype=float)
        xi = expr_T[i]
        Xj = expr_T[i+1:]
        if inputdesign == "hadamard":
            X_block = xi * Xj
        elif inputdesign == "concat":
            xi_block = np.tile(xi, (num_gene-1-i, 1))
            X_block[:, :num_sample] = xi_block
            X_block[:, num_sample:] = Xj
        elif inputdesign == "kron":
            for j in range(i+1, num_gene):
                xj = expr_T[j]
                X_block[j-i-1, :] = np.kron(xi, xj)

        if flag_1svm:
            # scores = clf.decision_function(X_block)
            scores = X_block @ w + b
        else:
            if clf1 is not None:
                # scores1 = clf1.decision_function(X_block)   
                scores1 = X_block @ w1 + b1
                scores = scores1

            if clf2 is not None:
                # scores2 = clf2.decision_function(X_block)
                scores2 = X_block @ w2 + b2
                if clf1 is not None:
                    scores[np.abs(scores2) > np.abs(scores1)] = -scores2[np.abs(scores2) > np.abs(scores1)]
                else:
                    scores = -scores2
        reg_estim[i, i+1:] = scores
        reg_estim[i+1:, i] = scores

    if flag_1svm:
        signs = np.sign(np.corrcoef(expr, rowvar=False))
        reg_estim = reg_estim * signs

    return reg_estim

def estimate_regulation_genie3(expr, num_gene_target=None, num_core=1, idx_regulator=None):
    """Estimate regulatory interactions using genie3 (random forest feature importance)."""
        
    num_sample = expr.shape[0]       
    num_gene = expr.shape[1]

    flag_all = False
    if num_gene_target is None:
        flag_all = True
        num_gene_target = num_gene

    reg_estim = np.zeros((num_gene, num_gene), dtype=float)
    reg_estim1 = np.zeros((num_gene, num_gene), dtype=float)
    print(f'Estimating regulation for {num_gene_target} target genes using GENIE3 with {num_core} cores...', flush=True)
    # target, temptf, feature_importances = estimate_regulation_genie3_one_target(0, expr, num_gene, idx_regulator) #debug
    results = Parallel(n_jobs=num_core, verbose=10, backend='loky')(
        delayed(estimate_regulation_genie3_one_target)(target, expr, num_gene, idx_regulator) for target in range(num_gene_target)
    )
    for target, temptf, feature_importances in results:
        reg_estim[temptf, target] = feature_importances
        reg_estim[target, temptf] = feature_importances
        reg_estim1[temptf, target] = feature_importances
    if flag_all:
        reg_estim = reg_estim1

    np.fill_diagonal(reg_estim, 0.0)
    signs = np.sign(np.corrcoef(expr, rowvar=False))
    reg_estim = reg_estim * signs

    return reg_estim
    
def estimate_regulation_genie3_one_target(target, expr, num_gene, idx_regulator=None):
    temptf = np.arange(num_gene) != target
    if idx_regulator is not None:
        temptf = temptf & np.isin(np.arange(num_gene), idx_regulator)
    X = expr[:, temptf]
    y = expr[:, target]

    rf = RandomForestRegressor(n_estimators=1000, n_jobs=1, max_features='sqrt', random_state=0)
    rf.fit(X, y)

    return target, temptf, rf.feature_importances_

def estimate_precision_by_nonsmooth_weighted_regression_single_add_gene(expr_reggiven, reg_given, expr, num_gene_reggiven, geneidx_add, method_glasso, hypparam, flag_Wupdate=False, flag_diagregularize=False, flag_parallel=False, flag_zerofix=False, list_theta_init=None, tol_prec=1e-4, tol_regress=1e-6):
    expr_now = np.hstack([expr_reggiven, expr[:, [geneidx_add]]])
    reg_given_now = np.zeros([num_gene_reggiven+1, num_gene_reggiven+1], dtype=float)
    reg_given_now[:num_gene_reggiven, :num_gene_reggiven] = reg_given.copy()

    isZ = None
    if flag_zerofix:
        isZ = np.zeros([num_gene_reggiven+1, num_gene_reggiven+1], dtype=bool)
        isZ[:num_gene_reggiven, :num_gene_reggiven] = (reg_given==0)

    theta, num_loop, obj_optim, num_optim_each, num_iter_each, msg_list = estimate_precision_by_nonsmooth_weighted_regression(
        expr_now, -reg_given_now, method_glasso, hypparam, list_theta_init=list_theta_init,
        flag_Wupdate=flag_Wupdate, flag_diagregularize=flag_diagregularize, flag_parallel=flag_parallel, isZ=isZ, tol_prec=tol_prec, tol_regress=tol_regress
    )

    partialcorr = -theta / np.sqrt(np.outer(np.diag(theta), np.diag(theta)))
    np.fill_diagonal(partialcorr, 0)
    reg_estim = partialcorr.copy()

    return geneidx_add, theta, reg_estim, num_optim_each, num_iter_each, msg_list  

# ================================
# Estimations (precision matrix)
# ================================
def estimate_precision_by_nonsmooth_weighted_regression(X, P_Q, method_glasso, hypparam, tol_prec=1e-4, tol_regress=1e-6, isZ=None, list_theta_init=None, flag_Wupdate=True, flag_diagregularize=False, flag_parallel=False, verbose=1):
    """
    Estimate the partial correlation matrix using nonsmooth weighted regression.
    """
    print_, msg_list = make_printer(flag_parallel=flag_parallel)
    if flag_parallel:
        P_Q_nodir = signed_edge_symmetrize(P_Q, verbose=False)
    else:
        P_Q_nodir = signed_edge_symmetrize(P_Q, verbose=verbose>0)
    flag_zerofix = isZ is not None
    if flag_zerofix:
        # both edges should be zero
        isZ_nodir = isZ & isZ.T
        np.fill_diagonal(isZ_nodir, False)
        if np.any((P_Q_nodir != 0) & isZ_nodir):
            raise ValueError("estimate_precision_by_nonsmooth_weighted_regression: Contradiction between P_Q and isZ.")
    else:
        isZ_nodir = None

    if flag_parallel:
        verbose = 1
    if verbose>0:
        print_("Estimating partial correlation matrix by nonsmooth weighted regression...")
    if method_glasso == "gammaguided":
        method_regress = "GGlasso"
        if not "alpha" in hypparam:
            hypparam["alpha"] = hypparam["gammamode"] * hypparam["beta"] + 1
        if not "gammamode" in hypparam:
            gammamode = (hypparam["alpha"]-1) / hypparam["beta"]
        else:
            gammamode = hypparam["gammamode"]
        func_objective = lambda theta, S: objective_graphgglasso(
            theta, S, P_Q_nodir, hypparam["gamma"], hypparam["alpha"], hypparam["beta"], 
            flag_diagregularize=flag_diagregularize, verbose=False, isZ=isZ_nodir)

    elif method_glasso == "weighted":
        method_regress = "wLasso"
        func_objective = lambda theta, S: objective_graphwlasso(
            theta, S, P_Q_nodir, hypparam["gamma"], hypparam["w"], 
            flag_diagregularize=flag_diagregularize, verbose=False, isZ=isZ_nodir)

    S = np.cov(X.T)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    try:
        model = GraphicalLasso(alpha=hypparam["gamma"]/2)
        model.fit(X)        
    except FloatingPointError:
        try: 
            print_("Theta calculation by graphical lasso failed due to numerical issues. Retrying with ridge regularization...")
            model = GraphicalLasso(alpha=hypparam["gamma"] / 2, covariance="precomputed")
            S_ridge = S + S[0,0]/1000 * np.eye(S.shape[0]) # After normalization, S[0,0] is same as other diagonal elements.
            model.fit(S_ridge)
        except FloatingPointError:
            print_("Non SPD result: the system is too ill-conditioned for this solver. The system is too ill-conditioned for this solver")
            print_("Returning 0 precision matrix.")
            theta = np.eye(S.shape[0])
            return theta, 0, 0, 0, 0, msg_list

    warnings.filterwarnings("default", category=ConvergenceWarning)

    theta_glasso = model.precision_ 
    theta_raw = np.linalg.inv(S)

    if method_glasso == "weighted" and hypparam["w"] == 1:
        if verbose>0:
            print_("Graphical LASSO solution is same as graphical weighted LASSO when w=1.")
        return theta_glasso, 0, 0, 0, 0, msg_list

    ## block coordinate descent like graphical lasso
    # calculate beta
    obj_optim = 0
    num_optim = 0
    num_iter = 0

    # calculate precision matrix
    if flag_diagregularize:
        W = S + np.eye(S.shape[0]) * hypparam["gamma"]
    else:
        W = S * 0.95 #scikit-learn uses 0.95
        np.fill_diagonal(W,np.diag(S))
    theta = np.linalg.inv(W)

    # listing of initial value candidates
    if list_theta_init is not None:
        list_theta_init = list_theta_init.copy()
        thetainitlist = [f"init{i}" for i in range(len(list_theta_init))]
        if verbose>0:
            print_("Using provided initial value for precision matrix.")
    else:
        list_theta_init = []
        thetainitlist = []

    if method_glasso == "weighted":        
        thetainitlist += ['gLASSO precision']
        list_theta_init += [theta_glasso.copy()]
        if verbose>0:
            print_("Using graphical LASSO precision matrix as initial value.")
    elif method_glasso == "gammaguided":
        # Selection of initial value
        thetainitlist_gnaisto = ['raw precision', 'raw precision with 1e-10', 'gLASSO precision', 'gLASSO precision with 1e-10']
        thetainitlist_gnaisto += ['raw precision with gammamode', 'gLASSO precision with gammamode']
        thetainitlist += thetainitlist_gnaisto
        
        for thetainit in thetainitlist_gnaisto:
            if thetainit.startswith('raw precision'):
                theta_temp = theta_raw.copy()
            elif thetainit.startswith('gLASSO precision'):
                theta_temp = theta_glasso.copy()
            if thetainit.endswith('with 1e-10'):
                # Adjust to small value where contradicting
                theta_temp[(theta_temp<=0) & (P_Q_nodir==1)] = 1e-10
                theta_temp[(theta_temp>=0) & (P_Q_nodir==-1)] = -1e-10
            elif thetainit.endswith('with gammamode'):
                # Adjust to gammamode
                theta_temp[P_Q_nodir>0] = gammamode
                theta_temp[P_Q_nodir<0] = -gammamode
            else:
                # Raw, but only for contradicting signs, adjust to gammamode
                theta_temp[(theta_temp<=0) & (P_Q_nodir==1)] = gammamode
                theta_temp[(theta_temp>=0) & (P_Q_nodir==-1)] = -gammamode
            list_theta_init.append(theta_temp)
            
        thetainitlist += ['safe initial value']
        theta_temp = np.zeros_like(theta)
        theta_temp[P_Q_nodir==1] = 1e-10
        theta_temp[P_Q_nodir==-1] = -1e-10
        np.fill_diagonal(theta_temp, np.diag(theta))
        list_theta_init.append(theta_temp)
    
    # Select the best initial value
    list_obj_init = {}
    bestobj_init = np.inf
    for thetainit, theta_now in zip(thetainitlist, list_theta_init):
        if np.all(np.linalg.eigvalsh(theta_now) > -1e-10):
            np.seterr(divide='ignore')
            obj_init = func_objective(theta_now, S)
            np.seterr(divide='warn')
        else:
            obj_init = np.inf

        list_obj_init[thetainit] = obj_init
        if obj_init < bestobj_init:
            bestobj_init = obj_init
            bestthetainit = thetainit
            theta = theta_now.copy()
    if verbose>0:
        print_(f"Best initial value: {bestthetainit} with objective {bestobj_init:.6f}")
        
    theta_first = theta.copy()
    W_first = W.copy()
    np.seterr(divide='ignore')
    obj_first = func_objective(theta, S)
    np.seterr(divide='warn')
    if verbose>0:
        print_(f"First Objective Value: {obj_first:.6f}")

    num_optim = 0
    num_iter = 0
    num_loop = 0
    list_obj = [obj_first]
    obj_now = obj_first
    obj_best = obj_first
    theta_best = theta_first.copy()
    W_best = W_first.copy()
    num_loop_best = -1
    list_diff_obj = []
    max_loop = 10
    while True:
        if verbose>0:
            print_(f"Loop {num_loop}...")
        obj_old = obj_now
        obj_optim = 0
        for i in range(X.shape[1]):
            if verbose>1:
                print_(f"Row {i}...")
            noti = np.arange(X.shape[1])
            noti = noti[noti != i]

            # beta_ref = np.linalg.solve(W[noti, :][:, noti], S[noti, i])
            XtY = -S[noti, i]
            P_Q_now = P_Q_nodir[noti, i]
            if flag_Wupdate:
                XtX = W[noti, :][:, noti]
                beta_init = theta[i, noti] / theta[i, i]
            else: 
                theta_noti_noti_inv = np.linalg.inv(theta[noti, :][:, noti])
                XtX = theta_noti_noti_inv
                beta_init = theta[i, noti] * W[i, i] 
            hypparam_now = hypparam.copy()
            hypparam_now.pop("gammamode", None)
            if method_glasso == "gammaguided":
                hypparam_now["alpha"] = W[i, i] * (hypparam_now["alpha"] - 1) + 1
            np.seterr(divide='ignore')
            if not flag_zerofix:
                beta, is_success, obj_optim_each, num_optim_each, num_iter_each = solve_nonsmooth_weighted_regression(
                    XtX, XtY, P_Q_now, method_regress, hypparam_now, A0=beta_init, flag_covmat=True, max_iter=5000, tol=tol_regress, verbose=verbose-1
                )
            else:
                isCalc = ~isZ_nodir[i, noti]
                beta_now, is_success, obj_optim_each, num_optim_each, num_iter_each = solve_nonsmooth_weighted_regression(
                    XtX[isCalc, :][:, isCalc], XtY[isCalc], P_Q_now[isCalc], method_regress, hypparam_now, A0=beta_init[isCalc], flag_covmat=True, max_iter=5000, tol=tol_regress, verbose=verbose-1
                )
                beta = np.zeros_like(beta_init)
                beta[isCalc] = beta_now

            np.seterr(divide='warn')
            
            if flag_Wupdate and is_success:
                # Update W
                cov_now = W[noti, :][:, noti] @ beta
                W[i, noti] = -cov_now
                W[noti, i] = -cov_now
                        
                # Update theta FIXME:add theta[noti, :][:, noti]
                cond_var = W[i, i] - np.dot(W[i, noti], beta)
                theta[i, i] = 1 / cond_var
                theta[i, noti] = beta * theta[i, i]
                theta[noti, i] = beta * theta[i, i]
                # theta = np.linalg.inv(W)
            elif is_success:
                # Update theta
                theta[i, noti] = beta / W[i, i]
                theta[noti, i] = beta / W[i, i]
                theta[i, i] = 1 / W[i, i] + theta[i, noti] @ theta_noti_noti_inv @ theta[noti, i]
                theta_test = theta.copy()
                np.fill_diagonal(theta_test, 0)

            num_optim += num_optim_each
            num_iter += num_iter_each
            
            np.seterr(divide='ignore')
            obj_now = func_objective(theta, S)
            np.seterr(divide='warn')

            list_obj.append(obj_now)
            if verbose>1:
                print_(f"Row {i} completed. Objective Value: {obj_now:.6f}")
                print_(f" Diff. from previous * w[i,i]: {(list_obj[-1] - list_obj[-2]) * W[i,i]}")

            if not np.isfinite(obj_now):
                break

        if verbose>0:
            print_(f"Loop {num_loop} completed. Objective value: {obj_now}, Number of optimizations: {num_optim}, Number of iterations: {num_iter}")

        if obj_now < obj_best:
            num_loop_best = num_loop
            obj_best = obj_now
            W_best = W.copy()
            theta_best = theta.copy()
            
        # Check for convergence
        diff_obj = (obj_now - obj_old) / (np.abs(obj_old) + 1e-8)
        list_diff_obj.append(diff_obj)
        if abs(diff_obj) < tol_prec and num_loop > 0:
            if verbose>0:
                print_(f"Convergence achieved with diff_obj: {abs(diff_obj):.6f} < tol: {tol_prec}.")
            break
        if diff_obj > 0:
            if verbose>0:
                print_(f"Objective increased with diff_obj: {diff_obj:.6f}.")
                print_(f"Resetting to best solution from loop {num_loop_best} ({obj_best})...")
            if num_loop_best==-1:
                if verbose>0:
                    print_("Warning: -1 -> Initial solution is best.")
            if obj_now > obj_first or np.isinf(obj_now):
                if verbose>0:
                    print_(f"Warning: Current solution ({obj_now}) is worse than the initial solution ({obj_first}).")
            obj_now = obj_best
            W = W_best.copy()
            theta = theta_best.copy()
            num_loop = num_loop_best
            break
                
        num_loop += 1
        if num_loop > max_loop:
            print_(f"Warning: Number of loops ({num_loop}) exceeds maximum ({max_loop}).")
            break
    if flag_Wupdate:
        theta = np.linalg.inv(W)
    else:
        W = np.linalg.inv(theta)

    flag_testplot = False
    if flag_testplot:
        print("PLOTTED")
        plt.figure()
        plt.plot(np.array(list_obj))
        diff_list_obj = np.diff(list_obj)
        plt.close()
    
    if verbose>0:
        print_("Precision matrix estimation completed.")
    return theta, num_loop, obj_now, num_optim, num_iter, msg_list

# ================================
# Evaluations
# ================================
def evaluate_regulation_semisupervise_estimation(expr, estim_method, eval_method, eval_method_opts, hypparam, **kwargs):
    """
    Estimate regulatory effects and evaluate based on the specified method.
    """
    print("")
    print(f"Evaluating model...")
    print(f"  method: {estim_method}")
    print(f"  evaluation method: {eval_method}")
    print(f"  evaluation options: {eval_method_opts}")
    # Evaluate based on the specified method
    if eval_method == "MAP_BlindGeneRegRanksum" :
        # Estimated regulation of blinded regulation of a gene, and calculate the correct ratio of the estimated regulation
        print("Evaluating MAP Blind Gene Regulation...")
        reg_known = kwargs.get('reg_known')
        i,j = np.nonzero(reg_known)
        geneinknown = np.unique(np.concatenate([i, j]))
        num_gene = expr.shape[1]
        
        if "unknowngenenum" in eval_method_opts.keys():
            seed = eval_method_opts.get("seed")
            random.seed(seed) 
            unknowngenenum = eval_method_opts["unknowngenenum"]
            geneinunknown = sorted(random.sample(range(len(geneinknown), num_gene), unknowngenenum))
            gene_eval = np.concatenate([geneinknown,geneinunknown])

        blindfold = eval_method_opts.get("blindfold", None)
        if blindfold is not None and len(geneinknown) != blindfold:
            seed = eval_method_opts.get("seed")
            random.seed(seed) 
            list_geneblind = geneinknown.copy()
            random.shuffle(list_geneblind)
            list_geneblind = [sorted(list_geneblind[i::5]) for i in range(5)]
        else:
            list_geneblind = [[x] for x in geneinknown]

        reg_allest_list = []
        reg_est = np.zeros((len(geneinknown), len(geneinknown)), dtype=float)
        rank_reg_est = np.zeros((len(geneinknown), len(geneinknown)), dtype=int)
        for geneblind in list_geneblind:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print(f"Blind Gene Regulation: Gene {geneblind}...")
            idxorder = list(range(num_gene))
            for i in geneblind:
                idxorder.remove(i)
                idxorder.insert(len(geneinknown)-1, i)

            reg_blindknown = reg_known.copy()
            reg_blindknown = reg_blindknown[idxorder, :][:, idxorder]
            for i in range(len(geneblind)):
                reg_blindknown[len(geneinknown)-len(geneblind)+i, :] = 0
                reg_blindknown[:, len(geneinknown)-len(geneblind)+i] = 0
            expr_now = expr[:, idxorder].copy()

            if "unknowngenenum" in eval_method_opts.keys():
                reg_blindknown = reg_blindknown[gene_eval, :][:, gene_eval].copy()
                expr_now = expr_now[:, gene_eval].copy()

            if estim_method in methodlist_undirected:
                reg_blindknown = signed_edge_symmetrize(reg_blindknown)
                
            if np.all(reg_blindknown == 0):
                print(" No known regulation. Skipped.")
                continue
            
            kwargs_blind = kwargs.copy()
            kwargs_blind['reg_known'] = reg_blindknown
            kwargs_blind['num_gene_reg_known'] = len(geneinknown)-len(geneblind)
            reg_allest, _ = estimate_regulation(expr_now, estim_method, hypparam, **kwargs_blind)
            reg_allest_list.append(reg_allest)

            reg_allest_temp = reg_allest[:, :len(geneinknown)-len(geneblind)].copy()
            reg_allest_temp[:len(geneinknown)-len(geneblind), :] = 0
            rank_reg_allest = np.argsort(np.argsort(np.abs(reg_allest_temp).ravel()))
            rank_reg_allest = rank_reg_allest.reshape(reg_allest_temp.shape)
            rank_reg_allest[reg_allest_temp == 0] = 0
            rank_reg_allest[reg_allest_temp != 0] = rank_reg_allest[reg_allest_temp != 0] - (len(geneinknown)-len(geneblind))**2 + 1 # exclude ranks of regulation between known genes

            colidx = geneinknown.copy()
            colidx = colidx[~np.isin(colidx, geneblind)]
            for i, g in enumerate(geneblind):
                reg_est[g, colidx] = reg_allest_temp[len(geneinknown)-len(geneblind)+i, :].copy()
                rank_reg_est[g, colidx] = rank_reg_allest[len(geneinknown)-len(geneblind)+i, :].copy()
        
        # Calculate the rank sum of the estimated regulation
        reg_est_flat = reg_est.flatten()
        rank_reg_est_flat = rank_reg_est.flatten()
        # if "nodir" in eval_method_opts:
        if eval_method_opts.get("nodir", False):
            reg_known_now = signed_edge_symmetrize(reg_known)
        else:  
            reg_known_now = reg_known.copy()
        reg_known_now = reg_known_now[geneinknown, :][:, geneinknown].copy()
        reg_known_flat = reg_known_now.flatten()
        evalscore = 0
        evalscore += np.sum(rank_reg_est_flat[(reg_known_flat > 0) & (reg_est_flat > 0)])
        evalscore += np.sum(rank_reg_est_flat[(reg_known_flat < 0) & (reg_est_flat < 0)])
        reg_eval = reg_allest_list
    else:
        raise ValueError(f"Unknown evaluation method: {eval_method}")
    isRegEstAllZero = np.all(reg_est==0)

    return evalscore, reg_eval, isRegEstAllZero

def evaluate_hyperparams(expr, estim_method, eval_method, eval_method_opts, hypparamgrid, **kwargs):
    """Evaluate hyperparameters based on the specified method."""

    print(f"Hyperparameter grid search for {estim_method}...")

    hypparamnames = hypparamgrid.keys()

    np_scores = []
    skip_hypparam = set()
    ranges = hypparamgrid.values()
    for hypparamval in itertools.product(*ranges):
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(f"Evaluating hyperparameters: {', '.join([f'{name}={val:.2g}' for name, val in zip(hypparamnames, hypparamval)])}...")

        if set(zip(hypparamnames, hypparamval)) & skip_hypparam:
            print("Skipped.")
            continue
        # Evaluate the model with the current hyperparameters
        np.seterr(divide='ignore')
        hypparam = dict(zip(hypparamnames, hypparamval))
        eval_score, reg_eval, isRegEstAllZero = evaluate_regulation_semisupervise_estimation(expr, estim_method, eval_method, eval_method_opts, hypparam, **kwargs)
        np.seterr(divide='warn')
        # Store the results
        row = hypparam.copy()
        row["score"] = eval_score
        np_scores.append(row)
        print("")
        print(f"Evaluation score ({eval_method}): {eval_score}")

        if isRegEstAllZero:
            print("All estimated regulations are zero.")
            if estim_method in ["NAISTO", "g2LASSO", "gNAISTO", "g3LASSO", "pergenewgLASSO", "wgLASSO", "gNAISTOz", "g3LASSOz", "pergenewgLASSOz", "wgLASSOz"]:
                print(f"Gamma {hypparam['gamma']} is too large or too small. Same gamma will be skipped in the next evaluations.")
                skip_hypparam.add(("gamma", hypparam['gamma']))
        print("//////////////////////////////////////////")

    ranking = pd.DataFrame(np_scores)
    best_row = ranking.loc[ranking['score'].idxmax()]
    best_hypparam = best_row.drop('score').to_dict()

    print("Best hyperparameters:")
    for param, value in best_hypparam.items():
        print(f" {param}: {value}")
    print(f"Best evaluation score: {best_row['score']}")

    return best_hypparam, ranking

def compare_estimated_and_correct_regulation(reg_estim, reg_correct):
    """Evaluate the estimated regulatory weights using AUPRC, AUROC, and F1 score."""
    # Flatten the matrices and create binary labels
    y_true = reg_correct.flatten()
    y_pred = reg_estim.flatten()

    if np.all(y_true == 0):
        print("Warning: No true regulations in the ground truth. AUPRC, AUROC, and F1 score cannot be computed.")
        AUPRC_sign, AUROC_sign, AUPRC, AUROC, F1 = np.nan, np.nan, np.nan, np.nan, np.nan
        precision_sign, recall_sign, tpr_sign, fpr_sign = np.nan, np.nan, np.nan, np.nan
    else:
        AUROC_sign, tpr_sign, fpr_sign = compute_roc_auc_signed(y_true, y_pred)
        AUPRC_sign, precision_sign, recall_sign = compute_auprc_signed(y_true, y_pred)

        AUPRC = average_precision_score(np.abs(y_true), np.abs(y_pred))
        AUROC = roc_auc_score(np.abs(y_true), np.abs(y_pred))
        F1 = f1_score(np.abs(y_true), np.abs(y_pred)>0)
        
    return AUPRC_sign, AUROC_sign, AUPRC, AUROC, F1, precision_sign, recall_sign, tpr_sign, fpr_sign

def compute_roc_auc_signed(y_true, y_score):
    order = np.argsort(-np.abs(y_score))
    y_true = y_true[order]
    y_score = y_score[order]

    P = np.sum(np.abs(y_true))
    N = len(y_true) - P

    fpr = [0.0]
    tpr = [0.0]
    tp = 0
    fp = 0

    scores_to_indices = defaultdict(list)
    for i, score in enumerate(np.abs(y_score)):
        scores_to_indices[score].append(i)

    for score in sorted(scores_to_indices.keys(), reverse=True):
        idxs = scores_to_indices[score]
        y_batch = y_score[idxs] * y_true[idxs]
        y_batch = y_batch[y_batch > 0]

        tp += len(y_batch)
        fp += len(idxs) - len(y_batch)

        fpr.append(fp / N)
        tpr.append(tp / P)

    auc = np.trapz(tpr, fpr)
    return auc, tpr, fpr

def compute_auprc_signed(y_true, y_score):
    order = np.argsort(-np.abs(y_score))
    y_true = y_true[order]
    y_score = y_score[order]

    P = np.sum(np.abs(y_true))
    tp = 0
    fp = 0

    precision = [1.0]  # sklearn
    recall = [0.0]

    scores_to_indices = defaultdict(list)
    for i, score in enumerate(np.abs(y_score)):
        scores_to_indices[score].append(i)

    for score in sorted(scores_to_indices.keys(), reverse=True):
        idxs = scores_to_indices[score]
        y_batch = y_score[idxs] * y_true[idxs]
        y_batch = y_batch[y_batch > 0]

        tp += len(y_batch)
        fp += len(idxs) - len(y_batch)

        prec = tp / (tp + fp)
        rec = tp / P

        precision.append(prec)
        recall.append(rec)

    # Step-wise integration
    auc = 0.0
    for i in range(1, len(recall)):
        delta_recall = recall[i] - recall[i - 1]
        auc += precision[i] * delta_recall

    return auc, precision, recall
