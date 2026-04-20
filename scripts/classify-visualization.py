import argparse
from pathlib import Path

import numpy as np
import cv2
from dash import Dash, dcc, html, Input, Output, ctx, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def edges_to_centers(bin_edges: np.ndarray) -> np.ndarray:
    return (bin_edges[1:] + bin_edges[:-1]) / 2


def flow_to_polar(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (H, W, 2) flow to magnitude and angle (degrees) arrays."""
    flow = flow.astype(np.float32)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag, np.degrees(ang)


def polar_to_rgb(mag: np.ndarray, ang_deg: np.ndarray) -> np.ndarray:
    """Build HSV-coded RGB image from precomputed magnitude and angle."""
    hsv = np.zeros((*mag.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang_deg / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sequence', type=str, help='path to sequence')
    args = parser.parse_args()

    sequence_path = Path(args.sequence)
    if not sequence_path.is_dir():
        raise ValueError(f"'{sequence_path}' is not a directory")

    data_file = sequence_path / "flow.npz"
    if not data_file.is_file():
        raise ValueError(f"'{data_file}' is not a file")

    npz = np.load(data_file)
    flows = npz['flows']
    timestamps = npz['timestamps']

    n_frames = flows.shape[0]
    N_BINS = 30

    # Global bin edges for consistent histogram axes across frames
    global_mags, global_angs = [], []
    for i in range(n_frames):
        data = np.transpose(flows[i, ...], (1, 2, 0))
        m, a = flow_to_polar(data)
        global_mags.append((m.min(), m.max()))
        global_angs.append((a.min(), a.max()))

    mag_bins = np.linspace(
        min(lo for lo, _ in global_mags),
        max(hi for _, hi in global_mags),
        num=N_BINS,
    )
    ang_bins = np.linspace(
        min(lo for lo, _ in global_angs),
        max(hi for _, hi in global_angs),
        num=N_BINS,
    )
    del global_mags, global_angs

    app = Dash(__name__)

    app.layout = html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "height": "100vh",
            "margin": "0",
            "padding": "10px",
            "boxSizing": "border-box",
        },
        children=[
            html.H2(
                "Optical Flow: Frame Explorer",
                style={"margin": "0 0 5px 0", "flexShrink": "0"},
            ),

            # Plots fill remaining space
            dcc.Graph(
                id="flow-graph",
                style={"flex": "1", "minHeight": "0"},
            ),

            # Store the selection range (persists across frame changes)
            dcc.Store(id="selection-store", data=None),

            # Controls pinned to bottom
            html.Div(
                style={
                    "flexShrink": "0",
                    "padding": "10px 0",
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "20px",
                },
                children=[
                    html.Div(
                        dcc.Slider(
                            id="frame-slider",
                            min=0,
                            max=n_frames - 1,
                            step=1,
                            value=0,
                            marks={
                                i: str(i)
                                for i in range(0, n_frames, max(1, n_frames // 10))
                            },
                            tooltip={"placement": "top", "always_visible": True},
                        ),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Label("Frame: "),
                            dcc.Input(
                                id="frame-input",
                                type="number",
                                min=0,
                                max=n_frames - 1,
                                step=1,
                                value=0,
                                debounce=True,
                                style={"width": "70px"},
                            ),
                        ],
                        style={"flexShrink": "0"},
                    ),
                    html.Button(
                        "Clear Selection",
                        id="clear-selection",
                        n_clicks=0,
                        style={"flexShrink": "0"},
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("frame-slider", "value"),
        Output("frame-input", "value"),
        Input("frame-slider", "value"),
        Input("frame-input", "value"),
    )
    def sync_controls(slider_val, input_val):
        trigger = ctx.triggered_id
        if trigger == "frame-input" and input_val is not None:
            val = int(np.clip(input_val, 0, n_frames - 1))
        else:
            val = slider_val if slider_val is not None else 0
        return val, val

    # Capture selection range from the heatmap subplot (axes x2/y2)
    @app.callback(
        Output("selection-store", "data"),
        Input("flow-graph", "selectedData"),
        Input("clear-selection", "n_clicks"),
    )
    def update_selection(selected_data, clear_clicks):
        trigger = ctx.triggered_id
        if trigger == "clear-selection":
            return None

        if not selected_data or not selected_data.get("range"):
            # selectedData can also carry "points"; use "range" for box select
            if selected_data and selected_data.get("points"):
                # Lasso / box: extract bounding box from points
                pts = selected_data["points"]
                xs = [p["x"] for p in pts]
                ys = [p["y"] for p in pts]
                return {
                    "mag_min": min(xs), "mag_max": max(xs),
                    "ang_min": min(ys), "ang_max": max(ys),
                }
            return no_update

        rng = selected_data["range"]
        # range keys are the axis ids: "x2" and "y2" for the second subplot
        x_key = next((k for k in rng if k.startswith("x")), None)
        y_key = next((k for k in rng if k.startswith("y")), None)
        if x_key and y_key:
            return {
                "mag_min": rng[x_key][0], "mag_max": rng[x_key][1],
                "ang_min": rng[y_key][0], "ang_max": rng[y_key][1],
            }
        return no_update

    # Main figure callback
    @app.callback(
        Output("flow-graph", "figure"),
        Input("frame-slider", "value"),
        Input("selection-store", "data"),
    )
    def update_figure(frame_idx: int, selection):
        data = np.transpose(flows[frame_idx, ...], (1, 2, 0))

        # Single polar conversion for both image and histogram
        mag, ang_deg = flow_to_polar(data)
        rgb = polar_to_rgb(mag, ang_deg)

        # Histogram in polar coordinates
        counts, mb, ab = np.histogram2d(
            mag.ravel(), ang_deg.ravel(), bins=[mag_bins, ang_bins]
        )
        log_counts = np.log10(
            counts, out=np.full_like(counts, np.nan), where=(counts > 0)
        )

        # Apply selection mask to the image
        if selection:
            mask = (
                (mag >= selection["mag_min"])
                & (mag <= selection["mag_max"])
                & (ang_deg >= selection["ang_min"])
                & (ang_deg <= selection["ang_max"])
            )
            display = rgb.copy()
            display[~mask] = (display[~mask] * 0.2).astype(np.uint8)
        else:
            display = rgb

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Dense Optical Flow", "Flow Distribution (Polar)"),
            horizontal_spacing=0.12,
        )

        fig.add_trace(
            go.Image(z=display),
            row=1, col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=edges_to_centers(mb),
                y=edges_to_centers(ab),
                z=log_counts.T,
                colorscale="Viridis",
                colorbar=dict(title="log₁₀(Count)", x=1.02),
            ),
            row=1, col=2,
        )

        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=1)

        fig.update_yaxes(title_text="Angle (°)", row=1, col=2)
        fig.update_xaxes(title_text="Magnitude (px)", row=1, col=2)

        # Enable box + lasso select on the heatmap subplot;
        # set dragmode to "select" so box select is the default tool.
        fig.update_layout(
            dragmode="select",
            margin=dict(l=40, r=40, t=40, b=20),
            # Ensure select tools appear in the modebar
            modebar=dict(
                add=["select2d", "lasso2d"],
            ),
        )

        return fig

    app.run(debug=True)


if __name__ == '__main__':
    main()
