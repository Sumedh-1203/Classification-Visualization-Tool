import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X_trans, y_true, X_original, y_pred, poly=None, scaler=None):
    
    # Axis limits from original data
    x_min, x_max = X_original[:,0].min()-1, X_original[:,0].max()+1
    y_min, y_max = X_original[:,1].min()-1, X_original[:,1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )

    # Grid in original space
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Apply polynomial transform if used
    if poly is not None:
        grid_trans = poly.transform(grid)
    else:
        grid_trans = grid

    # Apply scaling if used
    if scaler is not None:
        grid_trans = scaler.transform(grid_trans)

    # Predict decision boundary
    Z = model.predict(grid_trans).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Decision region
    ax.contourf(xx, yy, Z, alpha=0.3)

    # Scatter original data
    ax.scatter(X_original[:,0], X_original[:,1],
               c=y_true, edgecolor='k', s=40)

    # Highlight misclassified points
    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) > 0:
        ax.scatter(X_original[mis_idx,0], X_original[mis_idx,1],
                   facecolors='none', edgecolors='red',
                   s=100, linewidths=2, label="Misclassified")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right")

    return fig
