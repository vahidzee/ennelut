from __future__ import annotations

import torch
from matplotlib import pyplot as plt

from ennelut.data import LUT, generate_grid
from ennelut.model import ExactResFlow
from ennelut.utils import unnormalize


def visualize_needlemap(
    transform: ExactResFlow | LUT,
    save_path: str,
    grid_size: int,
    figsize: tuple[float, float] = (20, 20),
    quiver: bool = True,
    mesh_grid_size: int = 7,
    view_angle: tuple[float, float] = (22.5, 50),
    device: str = "cpu",
):
    torch.set_grad_enabled(False)

    # process the outputs grid
    inputs = generate_grid(grid_size=grid_size, meshgrid=True, device=device)
    if isinstance(transform, ExactResFlow):
        outputs = transform(inputs if transform.normalized else unnormalize(inputs))
        outputs = (unnormalize(outputs) if transform.normalized else outputs).cpu().numpy()
    else:
        outputs = unnormalize(transform(inputs)).cpu().numpy()
    inputs = unnormalize(inputs).cpu().numpy()
    grid = unnormalize(generate_grid(grid_size=grid_size, meshgrid=False, device="numpy"))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # prepare data for plotting
    x, y, z, u, v, w = [], [], [], [], [], []
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            for k in range(inputs.shape[2]):
                x.append(grid[i])
                y.append(grid[j])
                z.append(grid[k])
                u.append(outputs[i, j, k, 0] - inputs[i, j, k, 0])
                v.append(outputs[i, j, k, 1] - inputs[i, j, k, 1])
                w.append(outputs[i, j, k, 2] - inputs[i, j, k, 2])

    # Plotting
    if quiver:
        # q = ax.quiver(
        #     x,
        #     y,
        #     z,
        #     u,
        #     v,
        #     w,
        #     length=0.1,
        #     normalize=True,
        #     colors=outputs.reshape(-1, 3),
        #     linewidth=5,
        #     arrow_length_ratio=0.3,
        # )
        pass
    else:
        inputs = inputs.reshape(-1, 3)
        # sc = ax.scatter(inputs[:, 0], inputs[:, 1], inputs[:, 2], c=outputs.reshape(-1, 3), s=50, alpha=0.09)

    # rotate the view angle
    ax.view_init(*view_angle)

    # creating a mesh grid for better perspective
    # Drawing lines along the X, Y, and Z axes
    lattice_line_width = 2
    lattice = unnormalize(generate_grid(mesh_grid_size, meshgrid=False, device="numpy"))
    for i in range(lattice.shape[0]):
        i = lattice[i]
        for j in range(lattice.shape[0]):
            j = lattice[j]
            ax.plot(
                [i, i],
                [0, lattice[lattice.shape[0] - 1]],
                [j, j],
                color="black",
                linewidth=lattice_line_width,
                linestyle=":",
            )
            ax.plot(
                [j, j],
                [i, i],
                [0, lattice[lattice.shape[0] - 1]],
                color="black",
                linewidth=lattice_line_width,
                linestyle=":",
            )
            ax.plot(
                [0, lattice[lattice.shape[0] - 1]],
                [j, j],
                [i, i],
                color="black",
                linewidth=lattice_line_width,
                linestyle=":",
            )
    # turn off the axis
    ax.set_axis_off()

    # remove the color of the axis planes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # remove the outline of the axis planes
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")

    # save the figure with transparent background
    fig.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
