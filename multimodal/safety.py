import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import norm

class MMSafety():
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

    def compute_safe_control(self, xr, xh, ur_ref, thetas, belief, sigmas, return_slacks=False):
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
        
        # compute gamma(x,theta) for each theta
        gammas = np.zeros(thetas.shape[1])
        for i in range(thetas.shape[1]):
            sigma = sigmas[i]
            # TODO: solve as QCQP (for speed). for now, use scipy.optimize.minimize 
            obj = lambda x: -(x @ grad_phi_xh_flat) # so it's in the form obj(x) -> min
            const = lambda x: -(x.T @ sigma @ x) + 1 # so it's in the form const(x) >= 0
            res = minimize(obj, np.zeros(4), method="SLSQP", constraints={'type': 'ineq', 'fun': const})
            gammas[i] = res.fun
    
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

        # compute safety constraint per goal
        constraints = []
        for i in range(thetas.shape[1]):
            # computing constraint Lu <= S
            L = (grad_phi_xr @ self.r_dyn.g(xr)).flatten()
            uh_i = self.h_dyn.compute_control(xh, Ch.T @ thetas[:,[i]], Cr @ xr)
            S = -self.eta - self.lambda_r - (grad_phi_xr @ self.r_dyn.f(xr)).item() - (grad_phi_xh @ self.h_dyn.mean_dyn(xh, uh_i, thetas[:,[i]])).item() - gammas[i]*(k - slacks[i])
            
            constraints.append({'type': 'ineq', 'fun': lambda u: S - (L @ u).item()})

        # compute safe control
        obj = lambda u: np.linalg.norm(u - ur_ref)**2
        res = minimize(obj, ur_ref, method="SLSQP", constraints=constraints)
        
        # check if safe control is necessary (i.e. if safety constraint is active)
        d_dot = (d_p.T @ d_v) / d
        phi = self.dmin**2 + self.eta + self.lambda_r - d**2 - self.k_phi*d_dot.item()
        # compute safety index for each goal, where we additionally add gamma[i]*(k-slacks[i]) to phi. overall phi is the max of these
        for i in range(thetas.shape[1]):
            phi_i = self.dmin**2 + self.eta + self.lambda_r - d**2 - self.k_phi*d_dot.item() + gammas[i]*(k - slacks[i])
            phi = max(phi, phi_i)
        
        if phi > 0:
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
    
class SEASafety():
    def __init__(self, r_dyn, h_dyn, dmin=1, eta=0.1, lambda_r=0.1, k_phi=0.5):
        self.r_dyn = r_dyn
        self.h_dyn = h_dyn
        self.dmin = dmin
        self.eta = eta
        self.lambda_r = lambda_r
        self.k_phi = k_phi

    def compute_safe_control(self, xr, xh, ur_ref, thetas, belief, sigmas):
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
        xh_next = self.h_dyn.step_mean(xh, uh) # robot only has access to mean dynamics

        # computing constraint Lu <= S
        L = (grad_phi_xr @ self.r_dyn.g(xr)).flatten()
        lambda_SEA = (3) * np.sqrt(grad_phi_xh_flat @ sigma @ grad_phi_xh_flat.T) + self.lambda_r
        S = -self.eta - lambda_SEA - (grad_phi_xr @ self.r_dyn.f(xr)).item() - (grad_phi_xh @ xh_next).item()
        constraint = {'type': 'ineq', 'fun': lambda u: S - (L @ u).item()}

        # compute safe control
        obj = lambda u: np.linalg.norm(u - ur_ref)**2
        res = minimize(obj, ur_ref, method="SLSQP", constraints=constraint)

        # check if safe control is necessary (i.e. if safety constraint is active)
        d_dot = (d_p.T @ d_v) / d
        phi = self.dmin**2 + self.eta + lambda_SEA - d**2 - self.k_phi*d_dot.item()
        if phi > 0:
            # safety constraint is active
            return res.x[:,None], phi, True
        else:
            # safety constraint is inactive
            return ur_ref, phi, False

    def __call__(self, *args, **kwds):
        return self.compute_safe_control(*args, **kwds)