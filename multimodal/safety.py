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

    def compute_safe_control(self, xr, xh, ur_ref, thetas, belief, sigmas, return_slacks=False, time=None):
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

        # compute safety constraint per goal
        constraints = []
        Ls = []
        Ss = []
        for i in range(thetas.shape[1]):
            # computing constraint Lu <= S
            L = (grad_phi_xr @ self.r_dyn.g(xr)).flatten()
            uh_i = self.h_dyn.compute_control(xh, Ch.T @ thetas[:,[i]], Cr @ xr)
            S = -self.eta - self.lambda_r - (grad_phi_xr @ self.r_dyn.f(xr)).item() - (grad_phi_xh @ self.h_dyn.mean_dyn(xh, uh_i, 0)).item() - gammas[i]*(k - slacks[i])
            Ls.append(L)
            Ss.append(S)
            # constraints.append({'type': 'ineq', 'fun': lambda u: S - (L @ u).item()})
        
        # compute constraints
        # NOTE: if lambda's are created inside a loop, the variables will refer to the value of that *token* at *execution time*, so we need to hard-code 0,1,2 here
        fun = lambda u: Ss[0] - (Ls[0] @ u).item()
        constraints.append({'type': 'ineq', 'fun': fun})

        fun = lambda u: Ss[1] - (Ls[1] @ u).item()
        constraints.append({'type': 'ineq', 'fun': fun})

        fun = lambda u: Ss[2] - (Ls[2] @ u).item()
        constraints.append({'type': 'ineq', 'fun': fun})

        # # compute safe control
        obj = lambda u: np.linalg.norm(u - ur_ref)**2
        res = minimize(obj, ur_ref, method="SLSQP", constraints=constraints)
        constraints_satisfied = [c['fun'](res.x) for c in constraints]

        # check if safe control is necessary (i.e. if safety constraint is active)
        d_dot = (d_p.T @ d_v) / d
        # phi = self.dmin**2 + self.eta + self.lambda_r - d**2 - self.k_phi*d_dot.item()
        phi = self.dmin**2 - d**2 - self.k_phi*d_dot.item() # without discrete-time compensation terms
        # compute safety index for each goal, where we additionally add gamma[i]*(k-slacks[i]) to phi. overall phi is the max of these
        # for i in range(thetas.shape[1]):
        #     phi_i = self.dmin**2 + self.eta + self.lambda_r - d**2 - self.k_phi*d_dot.item() + gammas[i]*(k - slacks[i])
        #     phi = max(phi, phi_i)

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

    def compute_safe_control(self, xr, xh, ur_ref, thetas, belief, sigmas, return_slacks=False, time=None):
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