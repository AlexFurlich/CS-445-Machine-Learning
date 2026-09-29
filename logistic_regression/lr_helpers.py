""" Plotting helpers for the logistic regression lab.

author: Nathan Sprague
version: 9/29/26

"""

import numpy as np
import matplotlib.pyplot as plt

# Class colors: saturated enough to stand out against the pale shading.
CLASS_STYLES = [dict(color='#1f5fa8', marker='o', s=30, label='y = 0'),
                dict(color='#c0282d', marker='^', s=40, label='y = 1')]


def plot_1d(model, x, y, xlabel='x', ylabel='P(y = 1)'):
    """ Plot binary labels against a single feature, along with the
    model's estimated probability of the positive class. """
    jitter = np.random.default_rng(0).uniform(-.03, .03, size=y.shape)
    plt.plot(x, y + jitter, 'o', alpha=.3)
    xs = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    plt.plot(xs, model.predict_proba(xs)[:, 1], 'k-')
    plt.axhline(.5, color='gray', linestyle=':')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_probabilities(model, X, y, ax=None, title=None, equal_aspect=True):
    """ Plot two-dimensional data shaded by the model's estimated
    probability of the positive class.  The heavy line is the decision
    boundary, where that probability is .5.  The thin lines are where it
    is .1 and .9.  Class 0 is drawn as blue circles and class 1 as red
    triangles.  Pass equal_aspect=False when the two features have very
    different ranges.  Returns the shading so a colorbar can be
    attached. """
    if ax is None:
        ax = plt.gca()
    pad = 1.0
    x0s = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 200)
    x1s = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 200)
    grid_x0, grid_x1 = np.meshgrid(x0s, x1s)
    grid = np.column_stack([grid_x0.ravel(), grid_x1.ravel()])
    probs = model.predict_proba(grid)[:, 1].reshape(grid_x0.shape)

    shading = ax.contourf(grid_x0, grid_x1, probs,
                          levels=np.linspace(0, 1, 11), cmap='RdBu_r',
                          alpha=.3)
    ax.contour(grid_x0, grid_x1, probs, levels=[.1, .5, .9], colors='k',
               linewidths=[.5, 2, .5])
    for label, style in enumerate(CLASS_STYLES):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], edgecolors='white',
                   linewidths=.7, zorder=3, **style)
    if equal_aspect:
        ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title, fontsize=10)
    return shading
