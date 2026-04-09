"""
Creates a video of a sequence's optical flow estimates.
Example usage:
    python create-flow-video.py results/dsec/zurich_city_12_a
"""

import argparse

import numpy as np
import cv2


def flow_to_frames(flow: np.ndarray) -> np.ndarray:
    flow = flow.squeeze()
    _, h, w = flow.shape
    scaled_flow = np.rint((flow/2 + 2**7)).astype(np.uint8).transpose(1, 2, 0)
    flow_image = np.concatenate((scaled_flow, np.zeros((h, w, 1), dtype=np.uint8)), axis=-1)
    return flow_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sequence', type=str, help='path to sequence')
    parser.add_argument('--fps', type=int, default=30, help='fps of generated video')

    args = parser.parse_args()
    flows = np.load(f"{args.sequence}/flow.npy")
    frames = [ flow_to_frames(flows[i, ...]) for i in range(flows.shape[0]) ]
    height, width, layers = frames[0].shape

    print(f"Loaded {len(frames)} {width}x{height}x{layers} flow estimates")
    video = cv2.VideoWriter(f"{args.sequence}/flow.mp4", cv2.VideoWriter_fourcc(*'png '), args.fps, (width, height))
    for frame in frames: video.write(frame)
    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    main()