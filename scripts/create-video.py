"""
Creates a video of a sequence's optical flow estimates.
Example usage:
    python create-video.py results/dsec/zurich_city_12_a --flow --distribution
"""

import argparse
from pathlib import Path

import numpy as np
import cv2
from matplotlib import pyplot as plt, animation


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
    video = cv2.VideoWriter(path / "flow.mp4", cv2.VideoWriter_fourcc(*'png '), fps, (width, height))
    for frame in frames: video.write(frame)
    cv2.destroyAllWindows()
    video.release()


def create_distribution_video(path: Path, flows: np.ndarray, fps: float, ignore_zero: bool=True):
    bin_edges = np.linspace(-90, 90, 90)

    flow = flows[0, ...].reshape(2, -1)
    fig, ax = plt.subplots()
    im = ax.hist2d(flow[0, :], flow[1, :], bins=bin_edges, density=True)
    ax.set_xlabel("Horizontal Component (dps)")
    ax.set_ylabel("Vertical Component (dps)")
    ax.set_aspect(1.0)
    fig.colorbar(im[3], ax=ax)

    def plot(i):
        frame = flows[i, ...].reshape(2, -1)

        if ignore_zero:
            zero_mask = np.all(np.isclose(frame, 0, atol=1), axis=0)
            if not np.all(zero_mask): frame = frame[:, ~zero_mask]

        ax.clear()
        ax.hist2d(frame[0, :], frame[1, :], bins=bin_edges, density=True)
        ax.set_xlabel("Horizontal Component (dps)")
        ax.set_ylabel("Vertical Component (dps)")
        ax.set_title(f"Estimate {i} / {flows.shape[0]}")
        ax.set_aspect(1.0)

    ani = animation.FuncAnimation(fig, plot, frames=np.arange(flows.shape[0]), blit=False, interval=1e3 / fps)
    ani.save(path / "distribution.mp4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sequence', type=str, help='path to sequence')
    parser.add_argument('--flow', action='store_true', help='create a false-color video of the estimated flows')
    parser.add_argument('--distribution', action='store_true', help='create a video of the distribution of estimated flows')

    args = parser.parse_args()
    sequence_path = Path(args.sequence)
    if not sequence_path.is_dir(): raise ValueError(f"'{sequence_path}' is not a directory")

    data_file = sequence_path / "flow.npz"
    if not data_file.is_file(): raise ValueError(f"'{data_file}' is not a file")

    data = np.load(data_file)
    flows = data['flows']
    timestamps = data['timestamps']

    fps = 1e6 / np.diff(timestamps).mean()

    if args.flow:
        create_flow_video(sequence_path, flows, fps)
    if args.distribution:
        create_distribution_video(sequence_path, flows, fps)

if __name__ == '__main__':
    main()