import time
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy import abs as np_abs
from numpy import log as np_log

gammaZ = 1e4

# ===========================================
# GGlasso
# ===========================================
def objective_gglasso(A, P_Q, XtX, XtY, YtY, gamma, alpha, beta, datanum, verbose=False):    
    # Compute fit loss ||AX - Y||_F (Frobenius norm)
    fit_loss = A.T @ XtX @ A - 2 * XtY @ A + YtY

    # Regularization term using A and isP, isQ, isU
    abs_A = np_abs(A)
    log_abs_A = np_log(abs_A)

    abs_A_K = abs_A[P_Q!=0].tolist() # List length is small -> sum() is faster than np.sum() (calling np takes time)
    log_abs_A_K = log_abs_A[P_Q!=0].tolist()
    abs_A_U = abs_A[P_Q==0].tolist()
    regularization = -(alpha-1) * sum(log_abs_A_K) + beta * sum(abs_A_K) + gamma * sum(abs_A_U)

    if verbose:
        print(f"Fit Loss: {fit_loss}")
        print(f"Regularization: {regularization}")
        
    return fit_loss / datanum + regularization


def objective_wlasso(A, P_Q, XtX, XtY, YtY, gamma, w, datanum, verbose=False):    
    # Compute fit loss ||AX - Y||_F (Frobenius norm)
    fit_loss = A.T @ XtX @ A - 2 * XtY @ A + YtY

    # Regularization term using A and isP, isQ, isU
    abs_A = np_abs(A)

    abs_A_K = abs_A[P_Q!=0].tolist()
    abs_A_U = abs_A[P_Q==0].tolist()
    regularization = gamma * sum(abs_A_U) + gamma * w * sum(abs_A_K)

    if verbose:
        print(f"Fit Loss: {fit_loss}")
        print(f"Regularization: {regularization}")
        
    return fit_loss / datanum + regularization

def objective_graphwlasso(theta, S, P_Q, gamma, w, flag_diagregularize=False, verbose=False, isZ=None):
    sign, log_det_theta = np.linalg.slogdet(theta)
    if sign <= 0:
        return np.inf
    tr_Stheta = np.trace(S @ theta)
    fit_loss = -log_det_theta + tr_Stheta

    abs_theta = np_abs(theta)
    offdiag = ~np.eye(theta.shape[0], dtype=bool)

    abs_theta_K = abs_theta[(P_Q!=0) & offdiag].tolist()  # List length is small -> sum() is faster than np.sum()
    if isZ is None:
        abs_theta_U = abs_theta[(P_Q==0) & offdiag].tolist()    
        regularization = gamma * sum(abs_theta_U) + gamma * w * sum(abs_theta_K)
    else:
        abs_theta_U = abs_theta[(P_Q==0) & ~isZ & offdiag].tolist()
        abs_theta_Z = abs_theta[(P_Q==0) & isZ & offdiag].tolist()
        regularization = gamma * sum(abs_theta_U) + gamma * w * sum(abs_theta_K) + gammaZ * sum(abs_theta_Z)

    regularization = regularization / 2  # because theta is symmetric

    if flag_diagregularize:
        abs_theta_diag = abs_theta[~offdiag].tolist() 
        regularization = regularization + gamma * sum(abs_theta_diag)

    if verbose:
        print(f"Fit Loss: {fit_loss}")
        print(f"Regularization: {regularization}")

    return fit_loss + regularization

def objective_graphgglasso(theta, S, P_Q, gamma, alpha, beta, flag_diagregularize=False, verbose=False, isZ=None):
    # -log(det(theta)) + tr(Stheta)
    if np.any(P_Q * theta < 0):
        raise ValueError("Contradiction between P_Q and theta")
        
    sign, log_det_theta = np.linalg.slogdet(theta)
    if sign <= 0:
        return np.inf

    tr_Stheta = np.trace(S @ theta)
    fit_loss = -log_det_theta + tr_Stheta

    abs_theta = np_abs(theta)
    log_abs_theta = np_log(abs_theta)
    offdiag = ~np.eye(theta.shape[0], dtype=bool)

    abs_theta_K = abs_theta[(P_Q!=0) & offdiag].tolist()  # List length is small -> sum() is faster than np.sum()
    log_abs_theta_K = log_abs_theta[(P_Q!=0) & offdiag].tolist()
       
    if isZ is None:
        abs_theta_U = abs_theta[(P_Q==0) & offdiag].tolist()    
        regularization = -(alpha-1) * sum(log_abs_theta_K) + beta * sum(abs_theta_K) + gamma * sum(abs_theta_U)
    else:
        abs_theta_U = abs_theta[(P_Q==0) & ~isZ & offdiag].tolist()
        abs_theta_Z = abs_theta[(P_Q==0) & isZ & offdiag].tolist()
        regularization = -(alpha-1) * sum(log_abs_theta_K) + beta * sum(abs_theta_K) + gamma * sum(abs_theta_U) + gammaZ * sum(abs_theta_Z)

    regularization = regularization / 2  # because theta is symmetric

    if flag_diagregularize:
        abs_theta_diag = abs_theta[~offdiag].tolist() 
        regularization = regularization + gamma * sum(abs_theta_diag)


    if verbose:
        print(f"Fit Loss: {fit_loss}")
        print(f"Regularization: {regularization}")

    return fit_loss + regularization

def pgm_gglasso(x0, XtX, XtY, YtY, P_Q, beta, alpha, gamma, tau=1e-5, niter=10, tol=1e-8, backtracking=False, backtracking_decay=0.5, niterback=100):

    isP_Q = P_Q!=0
    isP = P_Q==1
    isQ = P_Q==-1
    one_pq = np.ones_like(P_Q[isP_Q])
    one_u  = np.ones_like(P_Q[~isP_Q])

    if backtracking:
        tau = 1.0

    t = 1.0
    x = x0.copy()
    y = x.copy()

    XtXx = XtX @ x
    pfg = np.dot(x, XtXx) - 2.0 * np.dot(x, XtY) + YtY
    absx = np.abs(x)
    logabsxsum_pq = np.dot(np.log(absx[isP_Q] + 1e-10), one_pq)
    pfg = pfg - logabsxsum_pq * (alpha - 1) + beta * np.dot(absx[isP_Q], one_pq) + gamma * np.dot(absx[~isP_Q], one_u)
            
    tolbreak = False
    incbreak = False

    for iiter in range(niter):
        xold = x.copy()

        XtXy = XtX @ y
        ygrad = 2 * (XtXy - XtY)
        absy_pq = np.abs(y[isP_Q]) + 1e-10 
        ygrad[isP_Q] -= (alpha - 1) * np.sign(y[isP_Q]) / absy_pq
                
        iiterback = 0
        while iiterback < niterback:
            x = y - tau * ygrad
            absx = np.abs(x)
            signx = np.sign(x)

            absx[isP_Q] = absx[isP_Q] - beta * tau
            absx[~isP_Q] = absx[~isP_Q] - gamma * tau
            absx[absx<0] = 0.0
            x = absx * signx
            temptf = (x<1e-10) & isP
            if any(temptf):
                x[temptf] = 1e-10
            temptf = (x>-1e-10) & isQ
            if any(temptf):
                x[temptf] = -1e-10

            xy = x - y
            norm_xy2 = np.dot(xy, xy)
                
            ft = np.dot(y, XtXy) - 2.0 * np.dot(y, XtY) + YtY
            ft -= np.dot(np.log(absy_pq), one_pq) * (alpha - 1)
            ft += np.dot(ygrad, xy) + 0.5 / tau * norm_xy2

            XtXx = XtX @ x
            pf = np.dot(x, XtXx) - 2.0 * np.dot(x, XtY) + YtY
            absx = np.abs(x)
            logabsxsum_pq = np.dot(np.log(absx[isP_Q] + 1e-10), one_pq)
            pf -= logabsxsum_pq * (alpha - 1)
            if pf <= ft:
                break
            tau *= backtracking_decay
            iiterback += 1

        told = t
        t = (1.0 + np.sqrt(1.0 + 4.0*t*t)) / 2.0   
        omega = ((told - 1.0) / t)
        y = x + omega * (x - xold)

        pfgold = pfg        
        pfg = pf + beta * np.dot(absx[isP_Q], one_pq) + gamma * np.dot(absx[~isP_Q], one_u)

        if pfgold == 0.0 and pfg == 0.0:
            pfgratio = 0.0
        else:
            pfgratio = np.abs(1.0 - pfg / pfgold)
        if pfgratio < tol:
            tolbreak = True
        elif pfg > pfgold:
            x = xold
            incbreak = True

        if tolbreak:
            break
        if incbreak:
            break

    return x, tolbreak, incbreak, iiter+1

def pgm_wlasso(x0, XtX, XtY, YtY, P_Q, w, gamma, tau=1e-5, niter=10, tol=1e-8, backtracking=False, backtracking_decay=0.5, niterback=100):

    isP_Q = P_Q!=0
    one_pq = np.ones_like(P_Q[isP_Q])
    one_u  = np.ones_like(P_Q[~isP_Q])
    gamma_w = gamma * w

    if backtracking:
        tau = 1.0

    t = 1.0
    x = x0.copy()
    y = x.copy()

    XtXx = XtX @ x
    pfg = np.dot(x, XtXx) - 2.0 * np.dot(x, XtY) + YtY
    absx = np.abs(x)
    pfg = pfg  + gamma_w * np.dot(absx[isP_Q], one_pq) + gamma * np.dot(absx[~isP_Q], one_u)
            
    tolbreak = False
    incbreak = False

    for iiter in range(niter):
        xold = x.copy()

        XtXy = XtX @ y
        ygrad = 2 * (XtXy - XtY)
                
        iiterback = 0
        while iiterback < niterback:
            x = y - tau * ygrad
            absx = np.abs(x)
            signx = np.sign(x)

            absx[isP_Q] = absx[isP_Q] - gamma_w * tau
            absx[~isP_Q] = absx[~isP_Q] - gamma * tau
            absx[absx<0] = 0.0
            x = absx * signx

            xy = x - y
            norm_xy2 = np.dot(xy, xy)
                
            ft = np.dot(y, XtXy) - 2.0 * np.dot(y, XtY) + YtY
            ft += np.dot(ygrad, xy) + 0.5 / tau * norm_xy2

            XtXx = XtX @ x
            pf = np.dot(x, XtXx) - 2.0 * np.dot(x, XtY) + YtY
            if pf <= ft:
                break
            tau *= backtracking_decay
            iiterback += 1

        told = t
        t = (1.0 + np.sqrt(1.0 + 4.0*t*t)) / 2.0   
        omega = ((told - 1.0) / t)
        y = x + omega * (x - xold)

        pfgold = pfg        
        absx = np.abs(x)
        pfg = pf + gamma_w * np.dot(absx[isP_Q], one_pq) + gamma * np.dot(absx[~isP_Q], one_u)

        if pfgold == 0.0 and pfg == 0.0:
            pfgratio = 0.0
        else:
            pfgratio = np.abs(1.0 - pfg / pfgold)
        if pfgratio < tol:
            tolbreak = True
        elif pfg > pfgold:
            x = xold
            incbreak = True

        if tolbreak:
            break
        if incbreak:
            break

    return x, tolbreak, incbreak, iiter+1

def solve_nonsmooth_weighted_regression(X, y, P_Q, method, hypparam, A0=None, flag_covmat=False, max_iter=50000, tol=1e-8, verbose=0):

    if verbose>0:
        print("")
    if method ==  "GGlasso":
        if verbose>0:
            print("Solving GGlasso...")
        gamma = hypparam.get('gamma')
        beta = hypparam.get('beta')  
        alpha = hypparam.get('alpha')
    elif method == "wLasso":
        if verbose>0:
            print("Solving wLasso...")
        gamma = hypparam.get('gamma')
        w = hypparam.get('w') # weight 

    # ================================
    # Initial Guess
    # ================================
    # Initialize A using linear regression
    if A0 is None:
        if verbose>0:
            print("Initializing...")
        if not flag_covmat:
            model = LinearRegression()
            model.fit(X, y)
            A0 = model.coef_
        else:
            A0 = np.linalg.solve(X, y) 

        if method == "GGlasso":
            gammamode = (alpha - 1) / beta
            for i in range(X.shape[1]):
                if P_Q[i]==1 and A0[i] < 1e-5:
                    A0[i] = gammamode
                elif P_Q[i]==-1 and A0[i] > -1e-5:
                    A0[i] = -gammamode
    else:
        if method == "GGlasso":
            for i in range(X.shape[1]):
                if P_Q[i]==1 and A0[i] <= 0:
                    raise ValueError("Initial value for P_Q given is wrong")
                elif P_Q[i]==-1 and A0[i] >= 0:
                    raise ValueError("Initial value for P_Q given is wrong")            
        
    # ================================
    # Solve the Optimization Problem
    # ================================
    tstart = time.time()

    # Use proximal gradient method with adaptive restart for optimization
    if method == "GGlasso":
        if not flag_covmat:
            datanum = X.shape[0]
            objective_partial = lambda A: objective_gglasso(A, P_Q, X.T @ X, X.T @ y, y.T @ y, gamma, alpha, beta, datanum, verbose=verbose>1)
        else:
            datanum = 1
            objective_partial = lambda A: objective_gglasso(A, P_Q, X, y, 0, gamma, alpha, beta, datanum, verbose=verbose>1)
    elif method == "wLasso":
        if not flag_covmat:
            datanum = X.shape[0]
            objective_partial = lambda A: objective_wlasso(A, P_Q, X.T @ X, X.T @ y, y.T @ y, gamma, w, datanum, verbose=verbose>1)
        else:
            datanum = 1
            objective_partial = lambda A: objective_wlasso(A, P_Q, X, y, 0, gamma, w, datanum, verbose=verbose>1)
        
    obj_initial = objective_partial(A0)
    if verbose>0:
        print(f"Initial Objective Value: {obj_initial}")

    list_method_pgm = ["fista"]
    A_est_old = A0
    
    # for method_pgm in list_method_pgm:
    num_optim = 0
    num_iter = 0
    while list_method_pgm:
    # until list_method_pgm is empty
        if verbose>1:
            print(f"FISTA{num_optim}")
        method_pgm = list_method_pgm.pop(0)
        tau = None # backtracking
        max_iter_each = 1000
        if method == "GGlasso":
            A_est, tolbreak, incbreak, num_iter_each = pgm_gglasso(
                x0=A_est_old, XtX=X, XtY=y, YtY=0, P_Q=P_Q, beta=beta, alpha=alpha, gamma=gamma,
                tau=tau, niter=max_iter_each, tol=tol, backtracking=True
                )
        elif method == "wLasso":
            A_est, tolbreak, incbreak, num_iter_each = pgm_wlasso(
                x0=A_est_old, XtX=X, XtY=y, YtY=0, P_Q=P_Q, w=w, gamma=gamma,
                tau=tau, niter=max_iter_each, tol=tol, backtracking=True
                )
        A_est_old = A_est.copy()
        num_optim += 1
        num_iter += num_iter_each
        obj_now = objective_partial(A_est)
        if verbose>1:
            print(f"Objective Value: {obj_now}")

        if tolbreak:
            if verbose>0:
                print("Tolerance condition met, stopping iterations")
            is_success = True
        if incbreak:
            if verbose>0:
                print("Increase of objective function")
            list_method_pgm.append(method_pgm)
        if num_iter > max_iter:
            print(f"Warning: Number of iteration is more than {max_iter}.")
            print("Stopping iterations.")
        elif num_iter_each >= max_iter_each:
            if verbose>0:
                print(f"Restart method")
            list_method_pgm.append(method_pgm)

    obj_final = obj_now
    if verbose>0:
        print(f"Final Objective Value: {obj_final}")
        print(f" Diff. from initial: {obj_final - obj_initial}")
    
    if obj_final > obj_initial:
        if verbose>0:
            print("Warning: Final objective value is greater than initial value.")
            print("Resetting A_est to initial guess.")
        is_success = False
        A_est = A0.copy()
    if not tolbreak:
        print("Warning: Tolerance condition not met.")
        print("Optimization Failed")
        is_success = False
    
    if verbose>0:
        print("")
    return A_est, is_success, obj_final, num_optim, num_iter  # Exclude the intercept term
