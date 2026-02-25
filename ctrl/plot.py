from matplotlib import patches, pyplot as plt, gridspec
from matplotlib.transforms import Affine2D
import torch
from IPython.display import HTML, display

__all__ = [
    'draw_car',
    'plot_τ',
    'plot_ctrl',
    'plot_xu',
    'plot_multi_controls',
    'plot_2_phase_planes',
]


def draw_car(ax, x, y, θ, width=0.4, length=1.0):
    rect = patches.Rectangle(
        (x, y - width / 2), 
        length,
        width,
        transform=Affine2D().rotate_around(*(x, y), θ) + ax.transData,
        alpha=0.8,
        fill=False,
        ec='grey',
    )
    ax.add_patch(rect)
    

def plot_τ(τ, y=None, car=False, ax_lims=None, title=None, fmt='o-'):
    """
    Plot trajectory of vehicles
    ax_lims is a tuple of two tuples ((x_lim_left, x_lim_right), (y_lim_bottom, y_lim_top))
    """
    if ax_lims is None:
        ax_lims = ((-1, 7), (-2, 2))
    plt.plot(τ[:,0], τ[:,1], fmt)
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel(r'$x \; [\mathrm{m}]$')
    plt.ylabel(r'$y \; [\mathrm{m}]$')
    
    if title: plt.title('Trajectory')
    if car:
        for xx, yy, θ in τ[:, :3]:
            draw_car(plt.gca(), xx, yy, θ)
    plt.locator_params(nbins=21, axis='x')
    plt.locator_params(nbins=12, axis='y', integer=True)

    if y is not None:
        if y.dim() == 1: y = y[None,...]
        plt.plot(y[:,0], y[:,1], 'x', markersize=20, markeredgewidth=2.5)


def plot_ctrl(u, title=None, legend=True, ylim=None):
    if title: plt.title('Control signals')
    N = len(u)
    plt.stem(torch.arange(N).numpy()-0.1, u[:,0], 'C1', markerfmt='C1o', basefmt='none')
    plt.stem(torch.arange(N).numpy()+0.1, u[:,1], 'C2', markerfmt='C2o', basefmt='none')
    plt.xlabel('$n$')
    if legend:
        plt.legend([r'$\varphi$', '$a$'], ncols=2, bbox_to_anchor=(0.4, 0))
    if ylim is not None:
        plt.ylim(ylim)
    else: plt.ylim([u.min().item() - .2, u.max().item() + .2])
    plt.locator_params(nbins=20, axis='x', integer=True)


def plot_xu(x, u):
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    plt.subplot(gs[0])
    plot_τ(x, car=True)
    plt.subplot(gs[1])
    plot_ctrl(u)
    plt.tight_layout()
    return gs


def plot_multi_controls(u, K):
    plt.figure(figsize=(10,2*K))
    ylim = u.min().item() - .2, u.max().item() + .2
    for k in range(K):
        plt.subplot(K,1,k+1)
        plot_ctrl(u[...,k], legend=k==K-1, ylim=ylim)


def plot_2_phase_planes(τ, y):
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plot_τ(τ[...,0], y)
    plt.subplot(1,2,2)
    plot_τ(τ[...,1], y)

