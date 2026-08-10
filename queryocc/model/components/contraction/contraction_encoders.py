import torch
from torch import Tensor, nn


class ProjectionMLP(nn.Module):
    def __init__(self, point_dimensions=2, hidden_features=[64, 32]):
        super().__init__()
        self.dim = point_dimensions

        layers = self._get_layers(hidden_features, point_dimensions)

        self.final_layer = nn.Linear(hidden_features[-1], point_dimensions)

        # Identity Initialization: Start with zero displacement
        self._initialize_weights()

        self.mlp_hidden = nn.Sequential(*layers)

    def _get_layers(self, hidden_features, point_dimensions):
        layers = []
        layers.append(nn.Linear(point_dimensions, hidden_features[0]))
        layers.append(nn.ReLU())

        for i in range(len(hidden_features) - 1):
            layers.append(nn.Linear(hidden_features[i], hidden_features[i + 1]))
            layers.append(nn.ReLU())
        return layers

    def _initialize_weights(self):
        nn.init.constant_(self.final_layer.weight, 0.0)
        nn.init.constant_(self.final_layer.bias, 0.0)

    def forward(self, base_coords):
        hidden_output = self.mlp_hidden(base_coords)
        displacement = self.final_layer(hidden_output)

        transformed_coords = base_coords + displacement

        return transformed_coords


class ProjectionMLPScaling(ProjectionMLP):
    def __init__(self, point_dimensions=2, hidden_features=[64, 32]):
        super().__init__(point_dimensions, hidden_features)

    def _initialize_weights(self):
        nn.init.constant_(self.final_layer.weight, 0.0)
        nn.init.constant_(self.final_layer.bias, 1.0)

    def forward(self, base_coords):
        hidden_output = self.mlp_hidden(base_coords)
        scaling_factor = self.final_layer(hidden_output)

        transformed_coords = base_coords * scaling_factor

        return transformed_coords


class ProjectionMLPHybrid(ProjectionMLP):
    def __init__(self, point_dimensions=2, hidden_features=[64, 32]):
        super().__init__(point_dimensions, hidden_features)
        self.final_layer = nn.Linear(hidden_features[-1], 2 * point_dimensions)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.constant_(self.final_layer.weight[..., : self.dim, :], 0.0)
        nn.init.constant_(self.final_layer.bias[..., : self.dim], 0.0)

        nn.init.constant_(self.final_layer.weight[..., self.dim :, :], 0.0)
        nn.init.constant_(self.final_layer.bias[..., self.dim :], 1.0)

    def forward(self, base_coords):
        hidden_output = self.mlp_hidden(base_coords)
        displacement = self.final_layer(hidden_output)[..., : self.dim]
        scaling_factor = self.final_layer(hidden_output)[..., self.dim :]
        transformed_coords = base_coords * scaling_factor
        transformed_coords = transformed_coords + displacement
        return transformed_coords


class ProjectionTanh(nn.Module):
    def __init__(self, scaling_factor=2.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x * self.scaling_factor)


class ProjectionMipNerf360(nn.Module):
    def __init__(self, point_cloud_range: Tensor, high_resolution_range: Tensor):
        super().__init__()
        # both inputs are tensors like [xmin, ymin, zmin, xmax, ymax, zmax]
        pc_lo = torch.tensor(point_cloud_range[:2], dtype=torch.float32)
        pc_hi = torch.tensor(point_cloud_range[3:5], dtype=torch.float32)

        hr_lo = torch.tensor(high_resolution_range[:2], dtype=torch.float32)
        hr_hi = torch.tensor(high_resolution_range[3:5], dtype=torch.float32)

        norm_extent = (pc_hi - pc_lo) / 2.0  # half-extent of full BEV (per-dim)
        hr_extent = (hr_hi - hr_lo) / 2.0  # half-extent of high-res box (per-dim)

        # R in normalized coordinates (so R <= 1 for hr inside pc_range)
        hr_norm = hr_extent / norm_extent  # shape (2,)

        self.register_buffer("hr_norm", hr_norm)
        self.register_buffer("norm_extent", norm_extent)

    def forward(self, coords: Tensor) -> Tensor:
        """
        coords: (..., 2) coordinates already normalized to the point_cloud_range
                using your _normalize_xy (so center -> 0, full range -> [-1,1]).
                coords may be outside [-1,1] if original points are outside the range.
        """
        abs_coords = coords.abs()  # (..., 2)
        R = self.hr_norm  # (2,)

        # per-dim mask where contraction should be applied. No early-out on mask.any():
        # that would force a GPU->CPU sync every call; torch.where below handles it.
        mask = abs_coords > R  # (..., 2)

        # We use a 1D-per-dim function that is continuous at r=R and maps infinity -> 1:
        # C(x) = sign(x) * ( 1 - B / |x| ), where B = R * (1 - R)
        # This satisfies C(R) = R and C(inf) = 1.
        eps = 1e-12
        abs_safe = abs_coords.clamp(min=eps)

        B = R * (1.0 - R)  # (2,) per-dim
        # ensure broadcasting over leading dims
        B = B.view(*([1] * (coords.dim() - 1)), 2)  # shape (... compatible, 2)

        contracted_vals = (1.0 - B / abs_safe) * coords.sign()
        # clamp to ensure we stay in [-1, 1] numerically
        contracted_vals = contracted_vals.clamp(min=-1.0, max=1.0)

        out = torch.where(mask, contracted_vals, coords)
        return out


if __name__ == "__main__":
    import plotly.graph_objects as go
    import torch

    from queryocc.model.components.contraction.point_contraction import PointContraction, PointContractionConfig

    def _viz(xy, xy_contracted, point_cloud_range, high_resolution_range, name="temp"):
        if isinstance(xy, torch.Tensor):
            xy = xy.detach().cpu().numpy()
            xy_contracted = xy_contracted.detach().cpu().numpy()
        fig = go.Figure()
        # Draw a line from each original point to the contracted point
        for i in range(xy.shape[1]):
            fig.add_annotation(
                x=xy_contracted[0, i, 0],
                y=xy_contracted[0, i, 1],
                ax=xy[0, i, 0],
                ay=xy[0, i, 1],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=4,  # small filled triangle
                arrowsize=1.2,  # scale of arrowhead
                arrowwidth=1.7,
                arrowcolor="black",
                opacity=0.9,
            )
        fig.add_trace(
            go.Scatter(
                x=xy[0, :, 0], y=xy[0, :, 1], mode="markers", marker=dict(symbol="square", size=15), name="Original"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xy_contracted[0, :, 0],
                y=xy_contracted[0, :, 1],
                mode="markers",
                marker=dict(symbol="circle", size=13),
                name="Contracted",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=high_resolution_range[[0, 3]][[0, 0, 1, 1, 0]],
                y=high_resolution_range[[1, 4]][[0, 1, 1, 0, 0]],
                mode="lines",
                line=dict(color="black", dash="dot", width=8.5),
                name="High Resolution Range",
            )
        )
        # axis equal

        # Remove axes and background and add grey grid
        fig.update_layout(
            xaxis=dict(nticks=10, showgrid=True, gridcolor="LightGray"),
            yaxis=dict(nticks=20, showgrid=True, gridcolor="LightGray"),
        )
        # Remove ticks
        fig.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        # Make all text larger
        fig.update_layout(
            font=dict(size=30),
        )
        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        fig.write_image(f"temp_{name}.png", width=1000, height=1000)
        # Save as svg
        fig.write_image(f"temp_{name}.svg", width=1000, height=1000)
        return fig

    # Uniformly distributed points between -10 and 10
    x = torch.linspace(-80, 80, 6)
    y = torch.linspace(-80, 80, 6)
    xy = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1)
    xy = xy.reshape(-1, 2)
    xy = xy.unsqueeze(0)
    point_cloud_range = torch.tensor([-50, -50, -10, 50, 50, 10])
    high_resolution_range = torch.tensor([-40, -40, -5, 40, 40, 5])

    # MipNerf360
    config = PointContractionConfig(encoder="mipnerf360", high_resolution_range=high_resolution_range)
    point_contraction = PointContraction(config, point_cloud_range)
    xy_contracted = point_contraction(xy, point_cloud_range)
    fig = _viz(xy, xy_contracted, point_cloud_range, high_resolution_range, name="mipnerf360")
    fig.show()

    # Displacement
    config = PointContractionConfig(dimensions=2, encoder="displacement", hidden_dims=(64, 32))
    point_contraction = PointContraction(config, point_cloud_range)
    xy_contracted = point_contraction(xy, point_cloud_range)
    _viz(xy, xy_contracted, point_cloud_range, high_resolution_range, name="displacement")

    # Scaling
    config = PointContractionConfig(dimensions=2, encoder="scaling", hidden_dims=(64, 32))
    point_contraction = PointContraction(config, point_cloud_range)
    xy_contracted = point_contraction(xy, point_cloud_range)
    _viz(xy, xy_contracted, point_cloud_range, high_resolution_range, name="scaling")

    # Hybrid
    config = PointContractionConfig(dimensions=2, encoder="hybrid", hidden_dims=(64, 32))
    point_contraction = PointContraction(config, point_cloud_range)
    xy_contracted = point_contraction(xy, point_cloud_range)
    _viz(xy, xy_contracted, point_cloud_range, high_resolution_range, name="hybrid")

    # Tanh
    config = PointContractionConfig(encoder="tanh", tanh_scaling_factor=2.0)
    point_contraction = PointContraction(config, point_cloud_range)
    xy_contracted = point_contraction(xy, point_cloud_range)
    _viz(xy, xy_contracted, point_cloud_range, high_resolution_range, name="tanh_2p0")

    # Tanh
    config = PointContractionConfig(encoder="tanh", tanh_scaling_factor=1.0)
    point_contraction = PointContraction(config, point_cloud_range)
    xy_contracted = point_contraction(xy, point_cloud_range)
    _viz(xy, xy_contracted, point_cloud_range, high_resolution_range, name="tanh_1p0")

    # Fancy plot
    max_range = 90
    line_width = 4
    x = torch.arange(5, 80, 5)
    y = torch.arange(5, 80, 5)

    colors = (x - x.min()) / (x.max() - x.min())
    import numpy as np
    from matplotlib.cm import get_cmap

    cmap = get_cmap("jet")
    colors = cmap(colors)[..., :3]
    colors = (colors * 255).astype(np.uint8)
    rgb_colors = [f"rgb({','.join(f'{c:.2f}' for c in rgb)})" for rgb in colors]

    x1 = -x
    x2 = -x
    x3 = x
    x4 = x

    y1 = -y
    y2 = y
    y3 = y
    y4 = -y

    xy1 = torch.stack([x1, y1], dim=-1)
    xy2 = torch.stack([x2, y2], dim=-1)
    xy3 = torch.stack([x3, y3], dim=-1)
    xy4 = torch.stack([x4, y4], dim=-1)

    num_rects = xy1.shape[0]
    rects = torch.zeros(num_rects, 5, 2)
    rects[:, 0, :] = xy1
    rects[:, 1, :] = xy2
    rects[:, 2, :] = xy3
    rects[:, 3, :] = xy4
    rects[:, 4, :] = xy1

    xy = torch.cat([xy1, xy2, xy3, xy4], dim=0)

    point_cloud_range = torch.tensor([-60, -60, -10, 60, 60, 10])
    high_resolution_range = torch.tensor([-32, -32, -5, 32, 32, 5])

    config = PointContractionConfig(encoder="mipnerf360", high_resolution_range=high_resolution_range)
    point_contraction = PointContraction(config, point_cloud_range)
    xy_contracted = point_contraction(xy, point_cloud_range)

    # Split xy_contracted into 4 parts
    xyc1, xyc2, xyc3, xyc4 = torch.split(xy_contracted, xy.shape[0] // 4, dim=0)

    # Create rectangles
    num_rects = xyc1.shape[0]
    crects = torch.zeros(num_rects, 5, 2)
    crects[:, 0, :] = xyc1
    crects[:, 1, :] = xyc2
    crects[:, 2, :] = xyc3
    crects[:, 3, :] = xyc4
    crects[:, 4, :] = xyc1

    fig = go.Figure()
    for i in range(num_rects):
        fig.add_trace(
            go.Scatter(
                x=list(crects[i, :, 0]),
                y=list(crects[i, :, 1]),
                mode="lines",
                line=dict(color=rgb_colors[i], width=line_width),
                name=f"Rect {i}",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=high_resolution_range[[0, 3]][[0, 0, 1, 1, 0]],
            y=high_resolution_range[[1, 4]][[0, 1, 1, 0, 0]],
            mode="lines",
            line=dict(color="black", dash="dot", width=line_width * 1.2),
            name="High Resolution Range",
        )
    )
    fig.update_layout(
        xaxis=dict(nticks=20, showgrid=True, gridcolor="LightGray"),
        yaxis=dict(nticks=40, showgrid=True, gridcolor="LightGray"),
    )
    fig.update_layout(xaxis=dict(range=[-max_range, max_range]), yaxis=dict(range=[-max_range, max_range]))
    # Set the x=0 and y=0 to also be light gray
    fig.update_layout(xaxis=dict(zerolinecolor="LightGray"), yaxis=dict(zerolinecolor="LightGray"))
    fig.update_layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    # axis equal
    fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(scaleanchor="x", scaleratio=1))
    fig.write_image("contract_fancy_contracted.png", width=1000, height=1000)
    fig.write_image("contract_fancy_contracted.svg", width=1000, height=1000)

    fig = go.Figure()
    for i in range(num_rects):
        fig.add_trace(
            go.Scatter(
                x=list(rects[i, :, 0]),
                y=list(rects[i, :, 1]),
                mode="lines",
                line=dict(color=rgb_colors[i], width=line_width),
                name=f"Rect {i}",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=high_resolution_range[[0, 3]][[0, 0, 1, 1, 0]],
            y=high_resolution_range[[1, 4]][[0, 1, 1, 0, 0]],
            mode="lines",
            line=dict(color="black", dash="dot", width=line_width * 1.2),
            name="High Resolution Range",
        )
    )
    fig.update_layout(
        xaxis=dict(nticks=20, showgrid=True, gridcolor="LightGray"),
        yaxis=dict(nticks=40, showgrid=True, gridcolor="LightGray"),
    )
    fig.update_layout(xaxis=dict(range=[-max_range, max_range]), yaxis=dict(range=[-max_range, max_range]))
    fig.update_layout(xaxis=dict(zerolinecolor="LightGray"), yaxis=dict(zerolinecolor="LightGray"))
    fig.update_layout(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(scaleanchor="x", scaleratio=1))
    fig.write_image("contract_fancy_original.png", width=1000, height=1000)
    fig.write_image("contract_fancy_original.svg", width=1000, height=1000)

    # Test contr
    import plotly.graph_objects as go
    import torch

    x = torch.linspace(-50, 50, 100)
    alpha = 0.8
    hr = 40.0

    x_norm = x / hr

    x_hr = x_norm * alpha

    b = (1 - alpha) / x_norm.abs()
    x_lr = x_norm.sign() * (1 - b)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=x_hr, mode="lines", name="HR"))
    fig.add_trace(go.Scatter(x=x, y=x_lr, mode="lines", name="LR"))
    fig.show()
