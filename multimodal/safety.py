import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import norm, chi2

class MMSafety():
    def __init__(self, r_dyn, h_dyn, dmin=1, eta=0.1, lambda_r=0.1, k_phi=0.5):
        self.r_dyn = r_dyn
        self.h_dyn = h_dyn
        self.dmin = dmin
        self.eta = eta
        self.lambda_r = lambda_r
        self.k_phi = k_phi
        self.slacks_prev = np.zeros(3)

    def slack_var_helper_(self, xh, xr, Ch, Cr, grad_phi_xh, thetas, gammas, k, slacks):
        diffs = np.zeros((thetas.shape[1], thetas.shape[1]))
        for i in range(thetas.shape[1]):
            for j in range(thetas.shape[1]):
                theta_i = thetas[:,[i]]
                uh_i = self.h_dyn.compute_control(xh, Ch.T @ theta_i, Cr @ xr)
                val1 = (grad_phi_xh @ self.h_dyn.mean_dyn(xh, uh_i, 0)).item() - gammas[i]*(k - slacks[i])
                # val1 = (grad_phi_xh @ self.h_dyn.step_mean(xh, uh_i)).item() - gammas[i]*(k - slacks[i])

                theta_j = thetas[:,[j]]
                uh_j = self.h_dyn.compute_control(xh, Ch.T @ theta_j, Cr @ xr)
                val2 = (grad_phi_xh @ self.h_dyn.mean_dyn(xh, uh_j, 0)).item() - gammas[j]*(k - slacks[j])
                # val2 = (grad_phi_xh @ self.h_dyn.step_mean(xh, uh_j)).item() - gammas[j]*(k - slacks[j])
                diffs[i,j] = val1 - val2
        return np.amax(diffs)
        # try boltzmann operator as continuous approximation of max (may be easier for optimizer)
        # alpha = 1
        # return np.sum(diffs * np.exp(alpha*diffs)) / np.sum(np.exp(alpha*diffs))

    def compute_safe_control(self, xr, xh, ur_ref, thetas, belief, sigmas, return_slacks=False, time=None, ax=None):
        """
        xr: robot state
        xh: human state
        ur_ref: robot reference control
        thetas: goal locations
        belief: probability over goals that human will reach
        sigmas: covariance of human's goal reaching distribution (one per goal)
        return_slacks: whether to return slack variables
        time: time at which to compute safe control (for debugging)
        ax: matplotlib axis to plot on (visualizing controls)
        """
        plot_controls = (ax is not None)
        # computing cartesian position difference
        Ch = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]]) # mapping human state to [x, y]
        Cr = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]]) # mapping robot state to [x, y]
        xy_h = Ch @ xh
        xy_r = Cr @ xr
        d_p = xy_r - xy_h # cartesian position difference
        d = np.linalg.norm(d_p)
        
        # computing cartesian velocity difference
        Vh = np.array([[0, 1, 0, 0],
                       [0, 0, 0, 1]]) # mapping human state to [x_dot, y_dot]
        vxy_h = Vh @ xh
        vxy_r = np.array([xr[2]*np.cos(xr[3]), xr[2]*np.sin(xr[3])]) # robot velocity in [x, y]
        d_v = vxy_r - vxy_h # cartesian velocity difference

        # compute grad_phi_xh 
        grad_phi_xh = np.zeros((1,4))
        grad_phi_xh += 2*(d_p.T @ Ch)
        grad_phi_xh += self.k_phi*((d_v.T @ Ch)/d + (d_p.T@self.h_dyn.B.T)/d - ((d_p.T@d_v) * d_p.T @ Ch)/(d**3))
        grad_phi_xh_flat = grad_phi_xh.flatten()

        # compute grad_phi_xr
        V = np.array([[0, 0, np.cos(xr[3]).item(), (-xr[2]*np.sin(xr[3])).item()],
                      [0, 0, np.sin(xr[3]).item(), (xr[2]*np.cos(xr[3])).item()]]) # p_dv_p_xr
        grad_phi_xr = np.zeros((1,4))
        grad_phi_xr += -2*(d_p.T @ Cr)
        grad_phi_xr += -self.k_phi*((d_v.T @ Cr)/d + (d_p.T @ V)/d - ((d_p.T@d_v) * d_p.T @ Cr)/(d**3))
        
        # compute gamma(x,theta) for each theta
        gammas = np.zeros(thetas.shape[1])
        for i in range(thetas.shape[1]):
            sigma = sigmas[i]
            # TODO: solve as QCQP (for speed). for now, use scipy.optimize.minimize 
            obj = lambda x: -(x @ grad_phi_xh_flat) # so it's in the form obj(x) -> min
            # TODO: can maybe be solved for noninvertible sigma if we take only invertible submatrix (since we don't care about x)
            const = lambda x: -(x.T @ sigma @ x) + 1 # so it's in the form const(x) >= 0
            res = minimize(obj, np.zeros(4), method="SLSQP", constraints={'type': 'ineq', 'fun': const})
            gammas[i] = -res.fun # negate objective value because we minimized -x^T grad_phi_xh, but wanted to maximize x^T grad_phi_xh
    
        # compute slack variables
        k = 3 # nominal 3-sigma bound
        epsilon = 0.003 # 99.7% confidence
        obj = lambda s: self.slack_var_helper_(xh, xr, Ch, Cr, grad_phi_xh, thetas, gammas, k, s) + np.linalg.norm(s)
        const = lambda s: (belief @ norm.cdf(k - s)) - (1-epsilon) # so it's in the form const(s) >= 0
        # const = lambda s: (belief @ chi2.cdf((k-s)**2, df=xh.shape[0])) - (1-epsilon) # multivariate normal CDF 
        const_lb = lambda s: s # force slack variables to be positive
        constraints = [{'type': 'ineq', 'fun': const}, {'type': 'ineq', 'fun': const_lb}]
        res = minimize(obj, self.slacks_prev, method="SLSQP", constraints={'type': 'ineq', 'fun': const})
        res = minimize(obj, np.zeros(thetas.shape[1]), method="SLSQP", constraints={'type': 'ineq', 'fun': const})
        # res = minimize(obj, np.zeros(thetas.shape[1]), method="COBYLA", constraints=constraints)
        # lb = 0
        # ub = np.inf
        # res = minimize(obj, np.zeros(thetas.shape[1]), method="trust-constr", constraints=NonlinearConstraint(const, lb, ub))
        slacks = res.x
        self.slacks_prev = slacks
        
        # if time > 0:
        #     import ipdb; ipdb.set_trace()

        # compute safety constraint per goal
        constraints = []
        Ls = []
        Ss = []
        for i in range(thetas.shape[1]):
            # computing constraint Lu <= S
            L = (grad_phi_xr @ self.r_dyn.g(xr)).flatten()
            uh_i = self.h_dyn.compute_control(xh, Ch.T @ thetas[:,[i]], Cr @ xr)
            S = -self.eta - self.lambda_r - (grad_phi_xr @ self.r_dyn.f(xr)).item() - (grad_phi_xh @ self.h_dyn.mean_dyn(xh, uh_i, 0)).item() - gammas[i]*(k - slacks[i])
            # S = -self.eta - self.lambda_r - (grad_phi_xr @ self.r_dyn.f(xr)).item() - (grad_phi_xh @ self.h_dyn.step_mean(xh, uh_i)).item() - gammas[i]*(k - slacks[i])
            Ls.append(L)
            Ss.append(S)
            # TODO: compute the tighest constraint and only use that one (they are all parallel constraints)
            # constraints.append({'type': 'ineq', 'fun': lambda u: S - (L @ u).item()})
        
        # compute constraints
        # NOTE: if lambda's are created inside a loop, the variables will refer to the value of that *token* at *execution time*, so we need to hard-code 0,1,2 here
        fun = lambda u: Ss[0] - (Ls[0] @ u).item()
        fun1 = lambda u: Ss[0] - Ls[0] @ u
        constraints.append({'type': 'ineq', 'fun': fun})

        fun = lambda u: Ss[1] - (Ls[1] @ u).item()
        fun2 = lambda u: Ss[1] - Ls[1] @ u
        constraints.append({'type': 'ineq', 'fun': fun})

        fun = lambda u: Ss[2] - (Ls[2] @ u).item()
        fun3 = lambda u: Ss[2] - Ls[2] @ u
        constraints.append({'type': 'ineq', 'fun': fun})

        # generate meshgrid of controls
        us = np.linspace(-30, 30, 500)
        U1, U2 = np.meshgrid(us, us)
        U = np.vstack((U1.flatten(), U2.flatten()))

        # compute if each point satisfies constraints
        c1_satisfied = fun1(U) >= 0
        c2_satisfied = fun2(U) >= 0
        c3_satisfied = fun3(U) >= 0

        if plot_controls:
            ax.cla()
            ax.scatter(U[0,c1_satisfied], U[1,c1_satisfied], c='b', label='c1 satisfied', alpha=0.1, s=2)
            ax.scatter(U[0,c2_satisfied], U[1,c2_satisfied], c='g', label='c2 satisfied', alpha=0.1, s=2)
            ax.scatter(U[0,c3_satisfied], U[1,c3_satisfied], c='purple', label='c3 satisfied', alpha=0.1, s=2)

            ax.scatter(ur_ref[0], ur_ref[1], c='r', label='ur_ref', s=100)

            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)

        # objective function
        obj = lambda u: np.linalg.norm(u - ur_ref)**2
        # constraints_satisfied = [c['fun'](res.x) for c in constraints]

        # check if safe control is necessary (i.e. if safety constraint is active)
        d_dot = (d_p.T @ d_v) / d
        # phi = self.dmin**2 + self.eta + self.lambda_r - d**2 - self.k_phi*d_dot.item()
        phi = self.dmin**2 - d**2 - self.k_phi*d_dot.item() # without discrete-time compensation terms
        # compute safety index for each goal, where we additionally add gamma[i]*(k-slacks[i]) to phi. overall phi is the max of these
        # for i in range(thetas.shape[1]):
        #     phi_i = self.dmin**2 + self.eta + self.lambda_r - d**2 - self.k_phi*d_dot.item() + gammas[i]*(k - slacks[i])
        #     phi = max(phi, phi_i)

        if phi > 0:
            # only solve if saefty constraint is active
            res = minimize(obj, ur_ref, method="SLSQP", constraints=constraints)
            # safety constraint is active
            ret = res.x[:,None], phi, True
        else:
            # safety constraint is inactive
            ret = ur_ref, phi, False
        if return_slacks:
            ret += (slacks,)
        return ret

    def __call__(self, *args, **kwds):
        return self.compute_safe_control(*args, **kwds)

class BaselineSafety():
    def __init__(self, r_dyn, h_dyn, dmin=1, eta=0.1, lambda_r=0.1, k_phi=0.5):
        self.r_dyn = r_dyn
        self.h_dyn = h_dyn
        self.dmin = dmin
        self.eta = eta
        self.lambda_r = lambda_r
        self.k_phi = k_phi
        self.slacks_prev = np.zeros(3)

    def compute_safe_control(self, xr, xh, ur_ref, thetas, belief, sigmas, return_slacks=False, time=None, ax=None):
        """
        xr: robot state
        xh: human state
        ur_ref: robot reference control
        thetas: goal locations
        belief: probability over goals that human will reach
        sigmas: covariance of human's goal reaching distribution (one per goal)
        return_slacks: whether to return slack variables
        time: time at which to compute safe control (for debugging)
        ax: matplotlib axis to plot on (visualizing controls)
        """
        plot_controls = (ax is not None)
        # computing cartesian position difference
        Ch = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]]) # mapping human state to [x, y]
        Cr = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]]) # mapping robot state to [x, y]
        xy_h = Ch @ xh
        xy_r = Cr @ xr
        d_p = xy_r - xy_h # cartesian position difference
        d = np.linalg.norm(d_p)
        
        # computing cartesian velocity difference
        Vh = np.array([[0, 1, 0, 0],
                       [0, 0, 0, 1]]) # mapping human state to [x_dot, y_dot]
        vxy_h = Vh @ xh
        vxy_r = np.array([xr[2]*np.cos(xr[3]), xr[2]*np.sin(xr[3])]) # robot velocity in [x, y]
        d_v = vxy_r - vxy_h # cartesian velocity difference

        # compute grad_phi_xh 
        grad_phi_xh = np.zeros((1,4))
        grad_phi_xh += 2*(d_p.T @ Ch)
        grad_phi_xh += self.k_phi*((d_v.T @ Ch)/d + (d_p.T@self.h_dyn.B.T)/d - ((d_p.T@d_v) * d_p.T @ Ch)/(d**3))
        grad_phi_xh_flat = grad_phi_xh.flatten()

        # compute grad_phi_xr
        V = np.array([[0, 0, np.cos(xr[3]).item(), (-xr[2]*np.sin(xr[3])).item()],
                      [0, 0, np.sin(xr[3]).item(), (xr[2]*np.cos(xr[3])).item()]]) # p_dv_p_xr
        grad_phi_xr = np.zeros((1,4))
        grad_phi_xr += -2*(d_p.T @ Cr)
        grad_phi_xr += -self.k_phi*((d_v.T @ Cr)/d + (d_p.T @ V)/d - ((d_p.T@d_v) * d_p.T @ Cr)/(d**3))
        
        # sample points from 3-sigma ellipse of each goal
        xh_all = []
        for i in range(len(sigmas)):
            sigma = sigmas[i]
            uh_i = self.h_dyn.compute_control(xh, Ch.T @ thetas[:,[i]], Cr @ xr)
            xh_i = self.h_dyn.step_mean(xh, uh_i)
            # based on math from https://freakonometrics.hypotheses.org/files/2015/11/distribution_workshop.pdf
            L = np.linalg.cholesky(sigma)
            x = np.random.normal(0,1,(200,4))
            z = np.linalg.norm(x, axis=1)
            z = z.reshape(-1,1).repeat(x.shape[1], axis=1)
            y = x/z * 3
            y_new = L @ y.T + xh_i
            xh_all.append(y_new)
            # for j in range(200):
            #     c2 = (y_new[:,[j]] - xh_i).T @ np.linalg.inv(sigma) @ (y_new[:,[j]] - xh_i)
            #     print(c2) # equals 9 (as desired, these points are of Mahalanobis distance 3 from the mean)
            # find which y_new maximizes grad_phi_xh @ y_new
            # ax.scatter(y_new[0,:], y_new[2,:], c="purple", s=1)

        xh_all = np.hstack(xh_all)
        Lf_phis = grad_phi_xh @ (xh_all - xh)/self.h_dyn.ts
        idx = np.argmax(Lf_phis)
        xh_constraint = xh_all[:,[idx]]

        xh_dot = (xh_constraint - xh) / self.h_dyn.ts

        # compute single safety constraint using this point
        L = (grad_phi_xr @ self.r_dyn.g(xr)).flatten()
        S = -self.eta - self.lambda_r - (grad_phi_xr @ self.r_dyn.f(xr)).item() - (grad_phi_xh @ xh_dot).item()
        const = lambda u: S - (L @ u).item()
        const1 = lambda u: S - L @ u
        obj = lambda u: np.linalg.norm(u - ur_ref)**2

        # check if safe control is necessary (i.e. if safety constraint is active)        
        d_dot = (d_p.T @ d_v) / d
        # phi = self.dmin**2 + self.eta + self.lambda_r - d**2 - self.k_phi*d_dot.item()
        phi = self.dmin**2 - d**2 - self.k_phi*d_dot.item() # without discrete-time compensation terms

        if plot_controls:
            # generate meshgrid of controls
            us = np.linspace(-30, 30, 500)
            U1, U2 = np.meshgrid(us, us)
            U = np.vstack((U1.flatten(), U2.flatten()))

            c_satisfied = const1(U) >= 0

            ax.cla()
            ax.scatter(U[0,c_satisfied], U[1,c_satisfied], c='purple', label='c1 satisfied', alpha=0.1, s=1)
            ax.scatter(ur_ref[0], ur_ref[1], c='r', label='ur_ref', s=100)

            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)

        if phi >= 0:
            # safety constraint is active
            res = minimize(obj, ur_ref, method="SLSQP", constraints={'type': 'ineq', 'fun': const})
            ret = res.x[:,None], phi, True
        else:
            # safety constraint is inactive
            ret = ur_ref, phi, False
        if return_slacks:
            ret += (np.zeros(3),)

        return ret

    def __call__(self, *args, **kwds):
        return self.compute_safe_control(*args, **kwds)

class MMLongTermSafety():
    def __init__(self, r_dyn, h_dyn, dmin=1, eta=0.1, lambda_r=0.1, k_phi=0.5):
        self.r_dyn = r_dyn
        self.h_dyn = h_dyn
        self.dmin = dmin
        self.eta = eta
        self.lambda_r = lambda_r
        self.k_phi = k_phi

    def slack_var_helper_(self, xh, xr, Ch, Cr, grad_phi_xh, thetas, gammas, k, slacks):
        diffs = np.zeros((thetas.shape[1], thetas.shape[1]))
        for i in range(thetas.shape[1]):
            for j in range(thetas.shape[1]):
                theta_i = thetas[:,[i]]
                uh_i = self.h_dyn.compute_control(xh, Ch.T @ theta_i, Cr @ xr)
                val1 = (grad_phi_xh @ self.h_dyn.mean_dyn(xh, uh_i, theta_i)).item() - gammas[i]*(k - slacks[i])

                theta_j = thetas[:,[j]]
                uh_j = self.h_dyn.compute_control(xh, Ch.T @ theta_j, Cr @ xr)
                val2 = (grad_phi_xh @ self.h_dyn.mean_dyn(xh, uh_j, theta_j)).item() - gammas[j]*(k - slacks[j])
                diffs[i,j] = val1 - val2
        return np.amax(diffs)
        # try boltzmann operator as continuous approximation of max (may be easier for optimizer)
        # alpha = 1
        # return np.sum(diffs * np.exp(alpha*diffs)) / np.sum(np.exp(alpha*diffs))

    def _compute_grad_phi_terms(self, xr, xh):
        # computing cartesian position difference
        Ch = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]]) # mapping human state to [x, y]
        Cr = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]]) # mapping robot state to [x, y]
        xy_h = Ch @ xh
        xy_r = Cr @ xr
        d_p = xy_r - xy_h # cartesian position difference
        d = np.linalg.norm(d_p)
        
        # computing cartesian velocity difference
        Vh = np.array([[0, 1, 0, 0],
                       [0, 0, 0, 1]]) # mapping human state to [x_dot, y_dot]
        vxy_h = Vh @ xh
        vxy_r = np.array([xr[2]*np.cos(xr[3]), xr[2]*np.sin(xr[3])]) # robot velocity in [x, y]
        d_v = vxy_r - vxy_h # cartesian velocity difference

        # compute grad_phi_xh 
        grad_phi_xh = np.zeros((1,4))
        grad_phi_xh += 2*(d_p.T @ Ch)
        grad_phi_xh += self.k_phi*((d_v.T @ Ch)/d + (d_p.T@self.h_dyn.B.T)/d - ((d_p.T@d_v) * d_p.T @ Ch)/(d**3))
        grad_phi_xh_flat = grad_phi_xh.flatten()

        # compute grad_phi_xr
        V = np.array([[0, 0, np.cos(xr[3]).item(), (-xr[2]*np.sin(xr[3])).item()],
                      [0, 0, np.sin(xr[3]).item(), (xr[2]*np.cos(xr[3])).item()]]) # p_dv_p_xr
        grad_phi_xr = np.zeros((1,4))
        grad_phi_xr += -2*(d_p.T @ Cr)
        grad_phi_xr += -self.k_phi*((d_v.T @ Cr)/d + (d_p.T @ V)/d - ((d_p.T@d_v) * d_p.T @ Cr)/(d**3))

        return grad_phi_xr, grad_phi_xh

    def compute_safe_control_old(self, xr, xh, ur_ref, thetas, belief, sigmas, return_slacks=False, time=None):
        """
        xr: robot state
        xh: human state
        ur_ref: robot reference control
        thetas: goal locations
        belief: probability over goals that human will reach
        sigmas: covariance of human's goal reaching distribution (one per goal)
        return_slacks: whether to return slack variables
        time: time at which to compute safe control (for debugging)
        """
        # computing cartesian position difference
        Ch = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]]) # mapping human state to [x, y]
        Cr = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]]) # mapping robot state to [x, y]
        xy_h = Ch @ xh
        xy_r = Cr @ xr
        d_p = xy_r - xy_h # cartesian position difference
        d = np.linalg.norm(d_p)
        
        # computing cartesian velocity difference
        Vh = np.array([[0, 1, 0, 0],
                       [0, 0, 0, 1]]) # mapping human state to [x_dot, y_dot]
        vxy_h = Vh @ xh
        vxy_r = np.array([xr[2]*np.cos(xr[3]), xr[2]*np.sin(xr[3])]) # robot velocity in [x, y]
        d_v = vxy_r - vxy_h # cartesian velocity difference

        # compute grad_phi_xh 
        grad_phi_xh = np.zeros((1,4))
        grad_phi_xh += 2*(d_p.T @ Ch)
        grad_phi_xh += self.k_phi*((d_v.T @ Ch)/d + (d_p.T@self.h_dyn.B.T)/d - ((d_p.T@d_v) * d_p.T @ Ch)/(d**3))
        grad_phi_xh_flat = grad_phi_xh.flatten()

        # compute grad_phi_xr
        V = np.array([[0, 0, np.cos(xr[3]).item(), (-xr[2]*np.sin(xr[3])).item()],
                      [0, 0, np.sin(xr[3]).item(), (xr[2]*np.cos(xr[3])).item()]]) # p_dv_p_xr
        grad_phi_xr = np.zeros((1,4))
        grad_phi_xr += -2*(d_p.T @ Cr)
        grad_phi_xr += -self.k_phi*((d_v.T @ Cr)/d + (d_p.T @ V)/d - ((d_p.T@d_v) * d_p.T @ Cr)/(d**3))
        
        # compute gamma(x,theta) for each theta
        gammas = np.zeros(thetas.shape[1])
        for i in range(thetas.shape[1]):
            sigma = sigmas[i]
            # TODO: solve as QCQP (for speed). for now, use scipy.optimize.minimize 
            obj = lambda x: -(x @ grad_phi_xh_flat) # so it's in the form obj(x) -> min
            const = lambda x: -(x.T @ sigma @ x) + 1 # so it's in the form const(x) >= 0
            res = minimize(obj, np.zeros(4), method="SLSQP", constraints={'type': 'ineq', 'fun': const})
            gammas[i] = -res.fun # negate objective value because we minimized -x^T grad_phi_xh, but wanted to maximize x^T grad_phi_xh
    
        # compute slack variables
        k = 3 # nominal 3-sigma bound
        epsilon = 0.003 # 99.7% confidence
        obj = lambda s: self.slack_var_helper_(xh, xr, Ch, Cr, grad_phi_xh, thetas, gammas, k, s)
        const = lambda s: (belief @ norm.cdf(k - s)) - (1-epsilon) # so it's in the form const(s) >= 0
        lb = 0
        ub = np.inf
        res = minimize(obj, np.zeros(thetas.shape[1]), method="COBYLA", constraints={'type': 'ineq', 'fun': const})
        # res = minimize(obj, np.zeros(thetas.shape[1]), method="trust-constr", constraints=NonlinearConstraint(const, lb, ub))
        slacks = res.x

        # computing long-term safety instead of one-step safety constraint
        n_rollout = 100 # how many rollouts to simulate per mode
        horizon = 5 # how many steps to simulate per rollout
        n_robot_init = 100 # how many robot initializations to sample
        # loop through each mode for the human
        # compute minimum distance per mode (given (k-s)sigma bound)
        min_dists = []
        for i, sigma in enumerate(sigmas):
            # compute k-sigma ellipse 
            eigenvalues, eigenvectors = np.linalg.eig(sigma)
            sqrt_eig = np.sqrt(eigenvalues)
            # use only xy components
            sqrt_eig = sqrt_eig[[0,2]]
            eigenvectors = eigenvectors[:,[0,2]]
            min_dists.append((k-slacks[i])*sqrt_eig[0])

        # check if robot is already in safe region
        is_safe = True
        for i in range(thetas.shape[1]):
            if np.linalg.norm(Cr @ xr - Ch @ xh) < min_dists[i]:
                is_safe = False
                break
            # computing constraint Lu <= S
            # L = (grad_phi_xr @ self.r_dyn.g(xr)).flatten()
            # uh_i = self.h_dyn.compute_control(xh, Ch.T @ thetas[:,[i]], Cr @ xr)
            # S = -self.eta - self.lambda_r - (grad_phi_xr @ self.r_dyn.f(xr)).item() - (grad_phi_xh @ self.h_dyn.mean_dyn(xh, uh_i, 0)).item() - gammas[i]*(k - slacks[i])
            # if (L @ ur_ref).item() > S:
            #     is_safe = False
            #     break

        if is_safe:
            # robot is already in safe region
            if return_slacks:
                return ur_ref, 0, False, slacks
            else:
                return ur_ref, 0, False

        safety_probs = [[] for _ in range(thetas.shape[1])]
        prob_safety_satisfied = []
        # sample robot initializations
        # sample x,y position from meshgrid between [-10,10]
        x = np.linspace(-10, 10, 10)
        y = np.linspace(-10, 10, 10)
        xx, yy = np.meshgrid(x, y)
        xr_sims = np.vstack((xx.flatten(), yy.flatten(), xr[2]*np.zeros(xx.shape).flatten(), xr[3]*np.zeros(xx.shape).flatten()))

        # xr_sims = xr + (2*np.random.rand(4, n_robot_init) - 1)
        # # only randomize x and y position
        # xr_sims[2,] = xr[2]
        # xr_sims[3,] = xr[3]
        for init_i in range(n_robot_init):
            xr_sim = xr_sims[:,[init_i]]
            safety_probs_i = []
            for theta_i in range(thetas.shape[1]):
                theta = thetas[:,[theta_i]]
                n_safe = 0
                for rollout in range(n_rollout): 
                    xh_sim = xh
                    safety_violated = False
                    for _ in range(horizon):
                        # need to compute goal-reaching reference control for robot at each step
                        # ur_ref_sim = self.r_dyn.compute_goal_control(xr_sim, thetas[:,0])

                        # grad_phi_xr, grad_phi_xh = self._compute_grad_phi_terms(xr_sim, xh_sim)
                        # # compute safety constraint per goal
                        # # computing constraint Lu <= S
                        # L = (grad_phi_xr @ self.r_dyn.g(xr_sim)).flatten()
                        # uh_sim = self.h_dyn.compute_control(xh_sim, Ch.T @ theta, Cr @ xr_sim)
                        # # TODO: probably have to re-solve for gammas and slacks here :(
                        # S = -self.eta - self.lambda_r - (grad_phi_xr @ self.r_dyn.f(xr_sim)).item() - (grad_phi_xh @ self.h_dyn.mean_dyn(xh, uh_sim, 0)).item() - gammas[theta_i]*(k - slacks[theta_i])

                        # if (L @ ur_ref_sim).item() > S:
                        #     safety_violated = True
                        #     break

                        # # update xr_sim and xh_sim
                        # xh_sim = self.h_dyn.step(xh_sim, uh_sim)
                        # xr_sim = self.r_dyn.step(xr_sim, ur_ref_sim)

                        uh_sim = self.h_dyn.compute_control(xh_sim, Ch.T @ theta, Cr @ xr_sim)
                        xh_sim = self.h_dyn.step(xh_sim, uh_sim)
                        # check if safety constraint is active (is the robot's state within (k-s)sigma bound for this mode?)
                        if np.linalg.norm(Cr @ xr_sim - Ch @ xh_sim) < min_dists[i]:
                            safety_violated = True
                            break
                    if not safety_violated:
                        n_safe += 1
                safety_probs[i].append(n_safe / n_rollout)
                safety_probs_i.append(n_safe / n_rollout)
            satisfied = belief @ np.array(safety_probs_i)
            prob_safety_satisfied.append(satisfied)

        if time > 10:
            import ipdb; ipdb.set_trace()
        xr_next = self.r_dyn.step(xr, ur_ref)
        # select all initializations that are safe with prob > 1-epsilon
        prob_safety_satisfied = np.array(prob_safety_satisfied)
        safe_init_idxs = np.where(prob_safety_satisfied > 1-epsilon)[0]
        safe_inits = xr_sims[:,safe_init_idxs]
        # pick the one that is closest to xr_next
        dists = np.linalg.norm(safe_inits - xr_next, axis=0)
        best_init = np.argmin(dists)
        xr_sim = safe_inits[:,[best_init]]
        # best_init = np.argmax(prob_safety_satisfied)
        # xr_sim = xr_sims[:,[best_init]]

        # NOTE: I don't think this is correct; it's possible that xr_sim itself & paths forward from there are safe but the path to xr_sim is not safe
        # compute robot's control s.t. it drives towards xr_sim
        ur = self.r_dyn.compute_goal_control(xr, xr_sim)
        
        if return_slacks:
            return ur, 0, True, slacks
        else:
            return ur, 0, True

    def _slack_var_helper(self, thetas, sigma0, k, slacks, horizon, dists):
        sigma_t = np.sqrt(np.tile(np.arange(1, horizon+1), (thetas.shape[1], 1)))
        sigma_t = (sigma_t.T * sigma0 * (k-slacks)).T
        is_safe = dists > sigma_t
        # return np.sum(dists - sigma_t)
        # return np.sum(is_safe) - (thetas.shape[1] * horizon)
        return sigma_t

    def _slack_var_obj(self, slacks):
        all_diffs = np.zeros((slacks.shape[0], slacks.shape[0]))
        for i in range(slacks.shape[0]):
            for j in range(slacks.shape[0]):
                all_diffs[i,j] = slacks[i] - slacks[j]
        return np.amax(all_diffs)

    def compute_safe_control(self, xr, xh, r_controller, thetas, belief, sigmas, return_slacks=False, time=None):
        # hyperparameters
        horizon = 5 # how many steps to simulate per rollout
        k = 3 # nominal 3-sigma bound
        epsilon = 0.003 # 99.7% confidence
        n_rollout = 100 # how many rollouts to simulate per mode
        n_init = 10 # how many new initial conditions to sample

        Ch = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]]) # mapping human state to [x, y]
        Cr = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]]) # mapping robot state to [x, y]

        # compute minimum distance per mode (given (k-s)sigma bound)
        one_sigma_dists = np.zeros(thetas.shape[1])
        for i, sigma in enumerate(sigmas):
            # compute k-sigma ellipse 
            eigenvalues, eigenvectors = np.linalg.eig(sigma)
            sqrt_eig = np.sqrt(eigenvalues)
            # use only xy components
            sqrt_eig = sqrt_eig[[0,2]]
            eigenvectors = eigenvectors[:,[0,2]]
            one_sigma_dists[i] = sqrt_eig[0]

        # rollout the human's mean dynamics and the robot's reference control for each mode
        all_dists = np.zeros((thetas.shape[1], horizon))
        for theta_i in range(thetas.shape[1]):
            theta = thetas[:,[theta_i]]
            xh_sim = xh
            xr_sim = xr
            for i in range(horizon):
                # compute controls for both agents
                uh_sim = self.h_dyn.compute_control(xh_sim, Ch.T @ theta, Cr @ xr_sim)
                ur_sim = r_controller(xr_sim, xh_sim)

                # update xr_sim and xh_sim
                xh_sim = self.h_dyn.step_mean(xh_sim, uh_sim)
                xr_sim = self.r_dyn.step(xr_sim, ur_sim)

                all_dists[theta_i, i] = np.linalg.norm(Cr @ xr_sim - Ch @ xh_sim)

        constraints = []
        # constraints.append({'type': 'ineq', 'fun': lambda s: self._slack_var_helper(thetas, one_sigma_dists, k, s, horizon, all_dists)})
        const = lambda s: (belief @ norm.cdf(k - s)) - (1-epsilon) # so it's in the form const(s) >= 0
        constraints.append({'type': 'ineq', 'fun': const})

        # compute slack variables
        obj = lambda s: self._slack_var_obj(s)
        res = minimize(obj, np.zeros(thetas.shape[1]), method="COBYLA", constraints=constraints)

        # check if we're safe with computed slack variables
        # is_safe = self._slack_var_helper(thetas, one_sigma_dists, k, res.x, horizon, all_dists) >= 0
        sigma_t = self._slack_var_helper(thetas, one_sigma_dists, k, res.x, horizon, all_dists)
        is_safe = np.all(all_dists > sigma_t)
        is_safe = is_safe and np.all(all_dists > self.dmin)
        # print(is_safe)

        if is_safe:
            if return_slacks:
                return r_controller(xr, xh), 0, False, res.x
            else:
                return r_controller(xr, xh), 0, False
        
        # sample robot controls
        # ur_init = (np.random.rand(2, n_init) * 100) - 50
        longterm_safety_probs = np.zeros(n_init)
        # sample robot states directly
        xr_init = xr + (10*np.random.rand(4, n_init) - 5)
        xr_init[2,] = 0
        xr_init[3,] = 0
        xr_init[:,[0]] = xr

        # loop through initializations and empirically compute safety probability
        for i_init in range(n_init):
            # ur_i = ur_init[:,[i_init]]
            # xr_sim_init = xr
            # if i_init != 0: # so we can compute long-term safety prob from initial state as comparison
            #     xr_sim_init = self.r_dyn.step(xr_sim, ur_i)
            xr_sim_init = xr_init[:,[i_init]]
            safety_probs = []
            for theta_i in range(thetas.shape[1]):
                theta = thetas[:,[theta_i]]
                n_safe = 0
                for _ in range(n_rollout):
                    xr_sim = xr_sim_init
                    xh_sim = xh
                    safety_violated = False
                    for i in range(horizon):
                        # compute controls for both agents
                        uh_sim = self.h_dyn.compute_control(xh_sim, Ch.T @ theta, Cr @ xr_sim)
                        ur_sim = r_controller(xr_sim, xh_sim)

                        # update xr_sim and xh_sim
                        xh_sim = self.h_dyn.step(xh_sim, uh_sim)
                        xr_sim = self.r_dyn.step(xr_sim, ur_sim)

                        # check if safety constraint is active (is the robot's state within sqrt(t)*(k-s)sigma bound for this mode?)
                        if np.linalg.norm(Cr @ xr_sim - Ch @ xh_sim) < sigma_t[theta_i, i]:
                            safety_violated = True
                            break
                    if not safety_violated:
                        n_safe += 1
                safety_probs.append(n_safe / n_rollout)
            longterm_safety_probs[i_init] = belief @ np.array(safety_probs)
        curr_safety_prob = longterm_safety_probs[0]
        # find direction of improvement that's closest to current xr
        improve_idxs = np.where(longterm_safety_probs > curr_safety_prob)[0]
        if len(improve_idxs) == 0:
            # no improvement possible
            # best_idx = np.argmax(longterm_safety_probs)
            import ipdb; ipdb.set_trace()
        else:
            # pick the one that is closest to xr
            dists = np.linalg.norm(xr_init[:,improve_idxs] - xr, axis=0)
            best_idx = improve_idxs[np.argmin(dists)]
            xr_goal = xr_init[:,[best_idx]]
            # compute robot's control s.t. it drives towards xr_goal (inversely proportional to improvement in safety prob)
            ur = self.r_dyn.compute_goal_control(xr, xr_goal)
            # ur = ur / (longterm_safety_probs[best_idx] - curr_safety_prob)
            
            ret = ur, 0, True
            if return_slacks:
                ret += (res.x,)
            return ret
        

    def __call__(self, *args, **kwds):
        return self.compute_safe_control(*args, **kwds)

class SEASafety():
    def __init__(self, r_dyn, h_dyn, dmin=1, eta=0.1, lambda_r=0.1, k_phi=0.5):
        self.r_dyn = r_dyn
        self.h_dyn = h_dyn
        self.dmin = dmin
        self.eta = eta
        self.lambda_r = lambda_r
        self.k_phi = k_phi

    def compute_safe_control(self, xr, xh, ur_ref, thetas, belief, sigmas, return_slacks=False, time=None, ax=None):
        """
        xr: robot state
        xh: human state
        ur_ref: robot reference control
        thetas: goal locations
        belief: probability over goals that human will reach
        sigmas: covariance of human's goal reaching distribution (one per goal)
        """
        # computing cartesian position difference
        Ch = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]]) # mapping human state to [x, y]
        Cr = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]]) # mapping robot state to [x, y]
        xy_h = Ch @ xh
        xy_r = Cr @ xr
        d_p = xy_r - xy_h # cartesian position difference
        d = np.linalg.norm(d_p)
        
        # computing cartesian velocity difference
        Vh = np.array([[0, 1, 0, 0],
                       [0, 0, 0, 1]]) # mapping human state to [x_dot, y_dot]
        vxy_h = Vh @ xh
        vxy_r = np.array([xr[2]*np.cos(xr[3]), xr[2]*np.sin(xr[3])]) # robot velocity in [x, y]
        d_v = vxy_r - vxy_h # cartesian velocity difference

        # compute grad_phi_xh 
        grad_phi_xh = np.zeros((1,4))
        grad_phi_xh += 2*(d_p.T @ Ch)
        grad_phi_xh += self.k_phi*((d_v.T @ Ch)/d + (d_p.T@self.h_dyn.B.T)/d - ((d_p.T@d_v) * d_p.T @ Ch)/(d**3))
        grad_phi_xh_flat = grad_phi_xh.flatten()

        # compute grad_phi_xr
        V = np.array([[0, 0, np.cos(xr[3]).item(), (-xr[2]*np.sin(xr[3])).item()],
                      [0, 0, np.sin(xr[3]).item(), (xr[2]*np.cos(xr[3])).item()]]) # p_dv_p_xr
        grad_phi_xr = np.zeros((1,4))
        grad_phi_xr += -2*(d_p.T @ Cr)
        grad_phi_xr += -self.k_phi*((d_v.T @ Cr)/d + (d_p.T @ V)/d - ((d_p.T@d_v) * d_p.T @ Cr)/(d**3))

        # compute xh(k+1|k) assuming most likely goal from our belief
        theta_idx = np.argmax(belief)
        theta = thetas[:,[theta_idx]]
        sigma = sigmas[theta_idx]
        uh = self.h_dyn.compute_control(xh, Ch.T @ theta, Cr @ xr)
        # xh_next = self.h_dyn.step_mean(xh, uh)
        # xh_dot = (xh_next - xh) / self.h_dyn.ts
        xh_dot = self.h_dyn.mean_dyn(xh, uh, 0)

        # computing constraint Lu <= S
        L = (grad_phi_xr @ self.r_dyn.g(xr)).flatten()
        lambda_SEA = (3/self.h_dyn.ts) * np.sqrt(grad_phi_xh_flat @ sigma @ grad_phi_xh_flat.T) + self.lambda_r
        S = -self.eta - lambda_SEA - (grad_phi_xr @ self.r_dyn.f(xr)).item() - (grad_phi_xh @ xh_dot).item()
        constraint = {'type': 'ineq', 'fun': lambda u: S - (L @ u).item()}

        # compute safe control
        obj = lambda u: np.linalg.norm(u - ur_ref)**2

        # check if safe control is necessary (i.e. if safety constraint is active)
        d_dot = (d_p.T @ d_v) / d
        # phi = self.dmin**2 + self.eta + lambda_SEA - d**2 - self.k_phi*d_dot.item()
        phi = self.dmin**2 - d**2 - self.k_phi*d_dot.item() # without discrete-time compensation terms
        if phi > 0:
            # safety constraint is active
            res = minimize(obj, ur_ref, method="SLSQP", constraints=constraint)
            ret = res.x[:,None], phi, True
        else:
            # safety constraint is inactive
            ret = ur_ref, phi, False

        if return_slacks:
            ret += (np.zeros(3),)

        return ret

    def __call__(self, *args, **kwds):
        return self.compute_safe_control(*args, **kwds)