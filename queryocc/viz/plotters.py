import numpy as np
import plotly.graph_objects as go
import torch
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from torch import Tensor

from queryocc.viz.plotter_utils import compute_bbox_corners_2d

EGO_VEHICLE = np.array(
    [
        [-0.5, -0.9],
        [-0.5, 0.9],
        [3.0, 0.9],
        [3.0, -0.9],
        [-0.5, -0.9],
    ]
)


COLORS_MAP = {
    "ped_crossing": [65, 105, 225],
    "divider": [128, 128, 0],
    "boundary": [255, 127, 80],
    "empty": [245, 245, 220],
    "vehicle": [0, 0, 0],
    "red_vehicle": [139, 0, 0],
    "ego_vehicle": [170, 170, 170],
}


def get_bev_features_figure(bev_features, invert_colors=False, color_by="PCA") -> go.Figure:
    feat = bev_features.detach().float().cpu().numpy()
    dim, height, width = feat.shape

    if color_by == "PCA":
        feat = feat / (np.linalg.norm(feat, axis=0, keepdims=True) + 1e-6)
        pca = PCA(n_components=3)

        feat_flat = feat.transpose(1, 2, 0).reshape(-1, dim)
        mask = np.sum(feat_flat, axis=-1) != 0
        feat_flat_masked = feat_flat[mask]

        reduced_vectors_pca_dense = pca.fit_transform(feat_flat_masked)

        reduced_vectors_pca_dense_colors = (reduced_vectors_pca_dense - np.min(reduced_vectors_pca_dense)) / (
            np.max(reduced_vectors_pca_dense) - np.min(reduced_vectors_pca_dense)
        )

        bev_feat_colors = np.ones((height * width, 3))
        bev_feat_colors[mask, :] = reduced_vectors_pca_dense_colors
        bev_feat_colors = bev_feat_colors.reshape(height, width, 3)
        bev_feat_colors = (255 * bev_feat_colors).astype(np.uint8)

    elif color_by == "PCA-sigmoid":

        def sigmoid(z):
            return torch.from_numpy(z).sigmoid().numpy()

        feat_sigmoid = sigmoid(feat)
        pca = PCA(n_components=3)

        feat_flat = feat_sigmoid.transpose(1, 2, 0).reshape(-1, dim)
        mask = np.sum(feat_flat, axis=-1) != 0
        feat_flat_masked = feat_flat[mask]

        reduced_vectors_pca_dense = pca.fit_transform(feat_flat_masked)

        reduced_vectors_pca_dense_colors = (reduced_vectors_pca_dense - np.min(reduced_vectors_pca_dense)) / (
            np.max(reduced_vectors_pca_dense) - np.min(reduced_vectors_pca_dense)
        )

        bev_feat_colors = np.ones((height * width, 3))
        bev_feat_colors[mask, :] = reduced_vectors_pca_dense_colors
        bev_feat_colors = bev_feat_colors.reshape(height, width, 3)
        bev_feat_colors = (255 * bev_feat_colors).astype(np.uint8)

    elif color_by == "norm":
        reduced_vectors_norm = np.linalg.norm(feat, axis=0)
        reduced_vectors_norm = (reduced_vectors_norm - np.min(reduced_vectors_norm)) / (
            np.max(reduced_vectors_norm) - np.min(reduced_vectors_norm)
        )
        cmap = get_cmap("turbo")
        image_array = cmap(reduced_vectors_norm)
        image_array = np.array(image_array)
        bev_feat_colors = (image_array * 255).astype(np.uint8)

    elif color_by == "norm_percentile_95":
        norms = np.linalg.norm(feat, axis=0)
        clipped = np.clip(norms, 0, np.percentile(norms, 95))
        normalized = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-6)
        cmap = get_cmap("turbo")
        image_array = cmap(normalized)
        image_array = np.array(image_array)
        bev_feat_colors = (image_array * 255).astype(np.uint8)

    elif color_by == "norm_percentile_99":
        norms = np.linalg.norm(feat, axis=0)
        clipped = np.clip(norms, 0, np.percentile(norms, 99))
        normalized = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-6)
        cmap = get_cmap("turbo")
        image_array = cmap(normalized)
        image_array = np.array(image_array)
        bev_feat_colors = (image_array * 255).astype(np.uint8)

    elif color_by == "norm_log_scale":
        norms = np.linalg.norm(feat, axis=0)
        norms = np.log1p(norms)  # or np.sqrt(norms)
        normalized = (norms - norms.min()) / (norms.max() - norms.min() + 1e-6)
        cmap = get_cmap("turbo")
        image_array = cmap(normalized)
        image_array = np.array(image_array)
        bev_feat_colors = (image_array * 255).astype(np.uint8)

    elif color_by == "norm_sqrt_scale":
        norms = np.linalg.norm(feat, axis=0)
        norms = np.sqrt(norms)
        normalized = (norms - norms.min()) / (norms.max() - norms.min() + 1e-6)
        cmap = get_cmap("turbo")
        image_array = cmap(normalized)
        image_array = np.array(image_array)
        bev_feat_colors = (image_array * 255).astype(np.uint8)

    elif color_by == "pca_1dim":
        feat = feat / (np.linalg.norm(feat, axis=0, keepdims=True) + 1e-6)
        pca = PCA(n_components=1)
        feat_flat = feat.transpose(1, 2, 0).reshape(-1, dim)
        mask = np.sum(feat_flat, axis=-1) != 0

        feat_flat_masked = feat_flat[mask]
        reduced_vectors_pca_dense = pca.fit_transform(feat_flat_masked)[..., 0]

        cmap = get_cmap("turbo")
        bev_feat_dense_colors = np.array(cmap(reduced_vectors_pca_dense))
        bev_feat_colors = np.ones((height * width, 4))
        bev_feat_colors[mask, :] = bev_feat_dense_colors
        bev_feat_colors = bev_feat_colors.reshape(height, width, 4)
        bev_feat_colors = (255 * bev_feat_colors).astype(np.uint8)

    elif color_by == "pca_1dim_sigmoid":

        def sigmoid(z):
            return torch.from_numpy(z).sigmoid().numpy()

        feat_sigmoid = sigmoid(feat)
        pca = PCA(n_components=1)
        feat_flat = feat_sigmoid.transpose(1, 2, 0).reshape(-1, dim)
        mask = np.sum(feat_flat, axis=-1) != 0

        feat_flat_masked = feat_flat[mask]
        reduced_vectors_pca_dense = pca.fit_transform(feat_flat_masked)[..., 0]

        cmap = get_cmap("turbo")
        bev_feat_dense_colors = np.array(cmap(reduced_vectors_pca_dense))
        bev_feat_colors = np.ones((height * width, 4))
        bev_feat_colors[mask, :] = bev_feat_dense_colors
        bev_feat_colors = bev_feat_colors.reshape(height, width, 4)
        bev_feat_colors = (255 * bev_feat_colors).astype(np.uint8)

    else:
        bev_feat_colors = feat

    if invert_colors:
        bev_feat_colors = 255 - bev_feat_colors

    bev_feat_colors = np.fliplr(bev_feat_colors)
    bev_feat_colors = np.flipud(bev_feat_colors)

    fig = go.Figure()
    fig.add_trace(go.Image(z=bev_feat_colors))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig


def get_probe_figure_occupancy_splat_bev(
    queries,
    predictions,
    point_cloud_range,
    num_samples_per_dim,
    pcd=None,
    bboxes=None,
    color_empty_with_fancy: bool = False,
) -> go.Figure:
    if isinstance(queries, Tensor):
        queries = queries.cpu().numpy()
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()

    if pcd is not None:
        if isinstance(pcd, Tensor):
            pcd = pcd.cpu().numpy()[0]
        pcd = _clean_pcd(pcd, point_cloud_range)

    if bboxes is not None:
        rectangles = {}
        # Make the bboxes from the format (class: {center: (x,y,z), size: (l,w,h), rotation: (3x3)}) to bounding boxes [x1,y1,x2,y2]
        for cls_, bbox in bboxes.items():
            center = bbox["center"][:, :2].clone()
            size = bbox["size"][:, :2].clone()
            rotation = bbox["rotation"][:, :2, :2].clone()
            rects = compute_bbox_corners_2d(center, size, rotation)
            rects = rects.cpu().numpy()
            # keep only rectangles that have atleast one point in the query range
            mask = ((rects[:, :, 0] < point_cloud_range[3]) & (rects[:, :, 0] > point_cloud_range[0])).any(axis=1)
            mask &= ((rects[:, :, 1] < point_cloud_range[4]) & (rects[:, :, 1] > point_cloud_range[1])).any(axis=1)

            rects = rects[mask, :, :]
            # clip the rectangles to the query range
            rects = np.clip(rects, point_cloud_range[0:2], point_cloud_range[3:5])
            rectangles[cls_] = rects

    queries = queries.reshape((*num_samples_per_dim, -1))
    predictions = predictions.reshape(num_samples_per_dim)

    height = num_samples_per_dim[0]
    width = num_samples_per_dim[1]

    # find the highest queries[:,:,2] value that is prediction
    max_z = queries[..., 2].max()
    min_z = queries[..., 2].min()
    # normalize height to 0-1
    queries[..., 2] = (queries[..., 2] - min_z) / (max_z - min_z)

    queries[~predictions] = 0
    heights = np.transpose(queries, (0, 1, 3, 2))[:, :, 2, :].max(axis=-1)

    # create heatmap with Turbo colorscale
    fig = go.Figure()

    cmap = get_cmap("turbo")
    image_array = cmap(heights)
    image_array = np.array(image_array)
    image_array = (image_array * 255).astype(np.uint8)

    if color_empty_with_fancy:
        heights_indices = np.transpose(queries, (0, 1, 3, 2))[:, :, 2, :].argmax(axis=-1)
        H, W, C = predictions.shape
        rows = np.arange(H)[:, None]  # shape (H, 1)
        cols = np.arange(W)[None, :]  # shape (1, W)
        preds_ = predictions[rows, cols, heights_indices]
        image_array[~preds_] = COLORS_MAP["empty"] + [255]

    fig.add_trace(go.Image(z=image_array[::-1, :, :]))

    # Draw ego vehicle
    scale_x = (point_cloud_range[4] - point_cloud_range[1]) / num_samples_per_dim[1]
    scale_y = (point_cloud_range[3] - point_cloud_range[0]) / num_samples_per_dim[0]

    # Rasterize the EGO_VEHICLE coordinates
    rasterized_ego_vehicle = -EGO_VEHICLE.copy()
    rasterized_ego_vehicle[:, 0] = (rasterized_ego_vehicle[:, 0] - point_cloud_range[0]) / scale_y
    rasterized_ego_vehicle[:, 1] = (rasterized_ego_vehicle[:, 1] - point_cloud_range[1]) / scale_x
    fig.add_trace(
        go.Scatter(
            x=rasterized_ego_vehicle[:, 1],
            y=rasterized_ego_vehicle[:, 0],
            mode="lines",
            fill="toself",
            fillcolor="rgba(0, 0, 0, 0.5)",
            line=dict(color="black"),
        )
    )

    if pcd is not None:
        rasterized_pcd = pcd.copy()
        rasterized_pcd[:, 0] = (rasterized_pcd[:, 0] - point_cloud_range[0]) / scale_x
        rasterized_pcd[:, 1] = (rasterized_pcd[:, 1] - point_cloud_range[1]) / scale_y

        fig.add_trace(
            go.Scatter(
                x=rasterized_pcd[:, 1],
                y=height - rasterized_pcd[:, 0],
                mode="markers",
                marker=dict(size=2, color="grey", opacity=0.25),
            )
        )

    if bboxes is not None:
        for cls_, rects in rectangles.items():
            rasterized_rects = rects.copy()
            rasterized_rects[:, :, 0] = (rasterized_rects[:, :, 0] - point_cloud_range[0]) / scale_x
            rasterized_rects[:, :, 1] = (rasterized_rects[:, :, 1] - point_cloud_range[1]) / scale_y

            for i in range(rasterized_rects.shape[0]):
                fig.add_trace(
                    go.Scatter(
                        x=rasterized_rects[i, [0, 1, 3, 2, 0], 1],
                        y=height - rasterized_rects[i, [0, 1, 3, 2, 0], 0],
                        mode="lines",
                        line=dict(color="black", width=1),
                    )
                )

    # Layout adjustments
    fig = _fix_bev_fig(fig, width, height)

    return fig


def get_probe_figure_3d_semantics_splat_bev(
    queries_in,
    occupied_in,
    colors_in,
    point_cloud_range,
    num_samples_per_dim,
    pcd=None,
    bboxes=None,
    label_mapper=None,
) -> go.Figure:
    if isinstance(queries_in, Tensor):
        queries = queries_in.cpu().clone().numpy()
    else:
        queries = queries_in.copy()

    if isinstance(occupied_in, Tensor):
        predictions = occupied_in.cpu().clone().numpy()
    else:
        predictions = occupied_in.copy()

    if isinstance(colors_in, Tensor):
        colors = colors_in.cpu().clone().numpy()
    else:
        colors = colors_in.copy()

    if pcd is not None:
        if isinstance(pcd, Tensor):
            pcd = pcd.cpu().numpy()[0]
        pcd = _clean_pcd(pcd, point_cloud_range)

    queries = queries.reshape((*num_samples_per_dim, -1))  # length, width, height, 3
    predictions = predictions.reshape(num_samples_per_dim)  # length, width, height
    colors = colors.reshape((*num_samples_per_dim, -1))  # length, width, height, 3

    length, width, height = num_samples_per_dim

    # Take the color of the highest prediction, namely the last index with true values in the last dimension
    z_indices = np.argmax(predictions[:, :, ::-1], axis=-1)
    no_occupied_mask = np.logical_not(predictions.any(axis=-1))
    x_coords, y_coords = np.indices((length, width))
    z_coords = height - 1 - z_indices
    bev_colors = colors[x_coords, y_coords, z_coords]
    bev_colors[no_occupied_mask] = COLORS_MAP["empty"]

    fig = go.Figure()
    fig.add_trace(go.Image(z=np.flipud(bev_colors)))
    # Draw ego vehicle
    scale_x = (point_cloud_range[4] - point_cloud_range[1]) / num_samples_per_dim[1]
    scale_y = (point_cloud_range[3] - point_cloud_range[0]) / num_samples_per_dim[0]

    # Rasterize the EGO_VEHICLE coordinates
    rasterized_ego_vehicle = -EGO_VEHICLE.copy()
    rasterized_ego_vehicle[:, 0] = (rasterized_ego_vehicle[:, 0] - point_cloud_range[0]) / scale_y
    rasterized_ego_vehicle[:, 1] = (rasterized_ego_vehicle[:, 1] - point_cloud_range[1]) / scale_x
    fig.add_trace(
        go.Scatter(
            x=rasterized_ego_vehicle[:, 1],
            y=rasterized_ego_vehicle[:, 0],
            mode="lines",
            fill="toself",
            fillcolor="rgba(192, 192, 192, 0.5)",
            line=dict(color="black"),
        )
    )

    if pcd is not None:
        rasterized_pcd = pcd.copy()
        rasterized_pcd[:, 0] = (rasterized_pcd[:, 0] - point_cloud_range[0]) / scale_x
        rasterized_pcd[:, 1] = (rasterized_pcd[:, 1] - point_cloud_range[1]) / scale_y

        fig.add_trace(
            go.Scatter(
                x=rasterized_pcd[:, 1],
                y=length - rasterized_pcd[:, 0],
                mode="markers",
                marker=dict(size=2, color="grey", opacity=0.25),
            )
        )

    # Layout adjustments
    fig = _fix_bev_fig(fig, width, length)
    return fig


def _clean_pcd(pcd, point_cloud_range):
    x_buffer = (point_cloud_range[3] - point_cloud_range[0]) * 0.07
    y_buffer = (point_cloud_range[4] - point_cloud_range[1]) * 0.07
    mask = (pcd[:, 0] > point_cloud_range[0] + x_buffer) & (pcd[:, 0] < point_cloud_range[3] - x_buffer)
    mask &= (pcd[:, 1] > point_cloud_range[1] + y_buffer) & (pcd[:, 1] < point_cloud_range[4] - y_buffer)
    pcd = pcd[mask]
    time_mask = abs(pcd[..., -1]) < 0.1
    pcd = pcd[time_mask]
    return pcd


def _fix_bev_fig(fig, width, height):
    fig.update_xaxes(autorange="reversed")
    fig.update_xaxes(range=[width, 0], showticklabels=False)
    fig.update_yaxes(range=[height, 0], showticklabels=False)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        width=width * 2,
        height=height * 2,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return fig
