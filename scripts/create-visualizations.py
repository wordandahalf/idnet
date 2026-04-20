"""
Creates a video of a sequence's optical flow estimates.
Example usage:
    python create-visualizations.py results/dsec/zurich_city_12_a --video-flow --video-distribution --plot-mean
"""

import argparse
from pathlib import Path

import numpy as np
import cv2
from matplotlib import pyplot as plt, animation, colors
from scipy import stats


def flow_to_frames(flow: np.ndarray) -> np.ndarray:
    flow = flow.squeeze()
    _, h, w = flow.shape
    scaled_flow = np.rint((flow/2 + 2**7)).astype(np.uint8).transpose(1, 2, 0)
    flow_image = np.concatenate((scaled_flow, np.zeros((h, w, 1), dtype=np.uint8)), axis=-1)
    return flow_image


def create_flow_video(path: Path, flows: np.ndarray, fps: float):
    frames = [flow_to_frames(flows[i, ...]) for i in range(flows.shape[0])]
    height, width, layers = frames[0].shape

    print(f"Loaded {len(frames)} {width}x{height}x{layers} flow estimates")
    video = cv2.VideoWriter(path / "flow.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames: video.write(frame)
    cv2.destroyAllWindows()
    video.release()


def create_distribution_video(path: Path, flows: np.ndarray, fps: float, ignore_zero: bool=False, idx=None):
    bin_edges = np.linspace(-90, 90, 90)

    single_frame = idx is not None
    idx = idx if single_frame is not None else 0
    flow = flows[idx, ...].reshape(2, -1)
    fig, ax = plt.subplots()
    im = ax.hist2d(flow[0, :], flow[1, :], bins=bin_edges, density=True, norm=colors.LogNorm(vmin=1e-7, vmax=1e-1))
    ax.set_xlabel("Horizontal Component (deg)")
    ax.set_ylabel("Vertical Component (deg)")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    ax.set_aspect(1.0)
    ax.set_aspect(1.0)
    fig.colorbar(im[3], ax=ax)

    if not single_frame:
        def plot(i):
            frame = flows[i, ...].reshape(2, -1)

            if ignore_zero:
                zero_mask = np.all(np.isclose(frame, 0, atol=1), axis=0)
                if not np.all(zero_mask): frame = frame[:, ~zero_mask]

            ax.clear()
            ax.hist2d(frame[0, :], frame[1, :], bins=bin_edges, density=True, norm=colors.LogNorm(vmin=1e-7, vmax=1e-1))
            ax.set_xlabel("Horizontal Component (deg)")
            ax.set_ylabel("Vertical Component (deg)")
            ax.set_title(f"Estimate {i} / {flows.shape[0]}")
            ax.set_xlim(-90, 90)
            ax.set_ylim(-90, 90)
            ax.set_aspect(1.0)

        ani = animation.FuncAnimation(fig, plot, frames=np.arange(flows.shape[0]), blit=False, interval=1e3 / fps)
        ani.save(path / "distribution.mp4")
    else:
        plt.savefig(path / f"distribution_{idx}.svg")


def plot_flow(path: Path, flows: np.ndarray, timestamps: np.ndarray):
    N, channels, height, width = flows.shape
    flows = flows.reshape(N, channels, height * width)
    mean_flows = np.mean(flows, axis=-1)
    mode_flows, _ = stats.mode(flows, axis=-1)

    plt.figure(figsize=(24, 4))
    plt.plot(timestamps[1:] / 1e6, mean_flows[:, 0], label='Horizontal (mean)', c='r')
    plt.plot(timestamps[1:] / 1e6, mode_flows[:, 0], label='Horizontal (mode)', c='r', ls='--')
    plt.plot(timestamps[1:] / 1e6, mean_flows[:, 1], label='Vertical (mean)', c='b')
    plt.plot(timestamps[1:] / 1e6, mode_flows[:, 1], label='Vertical (mode)', c='b', ls='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (deg)")
    plt.title("Flow Estimate Statistics")
    plt.xlim(0, timestamps[-1] / 1e6)
    plt.ylim(-90, 90)
    plt.legend(ncols=2)
    plt.tight_layout()
    plt.savefig(path / "statistics.svg")

    std_flows = np.std(flows.astype(np.float32), axis=-1)
    mean_std_flow = np.mean(std_flows, axis=0)

    plt.figure(figsize=(12, 4))
    plt.plot(timestamps[1:] / 1e6, std_flows[:, 0], label='Horizontal', c='r')
    plt.plot(timestamps[1:] / 1e6, std_flows[:, 1], label='Vertical', c='b')
    plt.title(f"Flow Spread\n({mean_std_flow[0]:.2f}, {mean_std_flow[1]:.2f})")
    plt.xlim(0, timestamps[-1] / 1e6)
    plt.ylim(0, 40)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Flow Spread (std.)")
    plt.savefig(path / "concentration.svg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sequence', type=str, help='path to sequence')
    parser.add_argument('--video-flow', action='store_true', help='create a false-color video of the estimated flows')
    parser.add_argument('--video-distribution', action='store_true', help='create a video of the distribution of estimated flows')
    parser.add_argument('--plot-flow', action='store_true', help='create a plot of the estimated flows')
    parser.add_argument('--frame-idx', type=int, default=None, help='create a plot of the estimated flows')

    args = parser.parse_args()
    sequence_path = Path(args.sequence)
    if not sequence_path.is_dir(): raise ValueError(f"'{sequence_path}' is not a directory")

    data_file = sequence_path / "flow.npz"
    if not data_file.is_file(): raise ValueError(f"'{data_file}' is not a file")

    data = np.load(data_file)
    flows = data['flows']
    timestamps = data['timestamps']

    fps = 1e6 / np.diff(timestamps).mean()

    if args.video_flow:
        create_flow_video(sequence_path, flows, fps)
    if args.video_distribution:
        create_distribution_video(sequence_path, flows, fps, idx=args.frame_idx)
    if args.plot_flow:
        plot_flow(sequence_path, flows, timestamps)

if __name__ == '__main__':
    main()