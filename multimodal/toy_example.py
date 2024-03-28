import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, chi2

def plot_safety_const(ax, Lx, grad_phi_x, f_x, color="red"):
    # plot half-space constraint induced by the safety constraint
    # Lx @ u <= -grad_phi_x @ f_x
    RHS = -grad_phi_x @ f_x
    halfspace = np.array([Lx[0], Lx[1], -RHS])

    # plot half-space
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal", adjustable="box")
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = halfspace[0]*X + halfspace[1]*Y + halfspace[2]
    ax.contour(X, Y, Z, levels=[0], colors=color)
    # fill in half-space
    # xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # fmt = {"color": "#8a7fad", "edgecolor": "b", "alpha": 0.3}
    # h = halfspace
    # sign = 1 
    # if h[1]== 0:
    #     xi = np.linspace(xlim[sign], -h[2]/h[0], 100)
    #     ax.fill_between(xi, ylim[0], ylim[1], **fmt)
    # else:
    #     ax.fill_between(x, (-h[2]-h[0]*x)/h[1], ylim[sign], **fmt)

def plot_gaussian(mean, cov):
    xs = np.linspace(0,10,100)
    ys = np.linspace(0,5,100)

    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i,j], Y[i,j]])
            Z[i,j] = multivariate_normal.pdf(x, mean, cov)

    # import ipdb; ipdb.set_trace()
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.contourf(X, Y, Z, cmap="Greens", alpha=0.5)
    k = 3.2 # mahalanobis distance for ~99.7% probability mass
    # plot k-sigma ellipse
    w, v = np.linalg.eig(cov)
    angle = np.arctan(v[1,0] / v[0,0])
    angle = np.degrees(angle)
    a = np.sqrt(w[0])*k
    b = np.sqrt(w[1])*k
    ellipse = plt.matplotlib.patches.Ellipse(mean, a, b, angle=angle, fill=False, edgecolor="black")
    # ax.add_patch(ellipse)
    # plt.show()


if __name__ == "__main__":
    mean = np.array([2.5, 2.5])
    cov = np.array([[1, 0.5], [0.5, 1]])

    plot_gaussian(mean, cov)

    # plt.show()
    Lx = np.array([1, 1])
    grad_phi_x = np.array([1, 1])
    f_x = mean
    fig, ax = plt.subplots()
    plot_safety_const(ax, Lx, grad_phi_x, f_x)
    plot_safety_const(ax, Lx, grad_phi_x, f_x + np.array([1,1]), color="blue")
    plt.show()