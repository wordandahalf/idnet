import argparse
import types
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from pathlib import Path, PurePath
from typing import Tuple, List, Dict, Optional

import numpy
import torch
import h5py
import cv2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from idn.loader.loader_dsec import rec_train_collate, train_collate, assemble_dsec_test_set, EventSlicer, Sequence
from idn.model.idedeq import IDEDEQIDO, RecIDE
from idn.utils.dsec_utils import RepresentationType, VoxelGrid
from idn.utils.helper_functions import move_batch_to_cuda
from idn.utils.model_utils import get_model_by_name

MODEL_TYPE = { "id-4x": "IDEDEQIDO", "id-8x": "IDEDEQIDO", "tid": "RecIDE" }

MODEL_CONFIG = {
    "IDEDEQIDO": {
        "deblur": True, "add_delta": True, "update_iters": 4, "zero_init": True, "deq_mode": False, "input_flowmap": True
    },
    "RecIDE": {
        "update_iters": 1, "pred_next_flow": True,
    }
}

MODEL_CONFIG_OVERRIDES = {
    "id-8x": { "hidden_dim": 96, "downsample": 8, "pretrain_ckpt": "idn/checkpoint/id-8x.pt" },
    "id-4x": { "hidden_dim": 128, "downsample": 4, "pretrain_ckpt": "idn/checkpoint/id-4x.pt" },
    "tid": { "pretrain_ckpt": "idn/checkpoint/tid.pt" },
}

DATASET_CONFIG = {
    "downsample_ratio": 1,
    "concat_seq": False,
    "num_voxel_bins": 15,
    "dataset_name": "dsec",
}

DATASET_CONFIG_OVERRIDES = {
    "IDEDEQIDO": {},
    "RecIDE": { "recurrent": True, "sequence_length": 4 },
}

DATALOADER_CONFIG = {
    "batch_size": 1,
    "shuffle": False,
    "num_workers": 0,
    "pin_memory": True,
}

@dataclass
class Timebase:
    start: Optional[int]
    end:  Optional[int]
    dt:    int

class EcdSlicer(EventSlicer):
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = np.array(self.h5f['events/{}'.format(dset_str)])

        # ECD stores timestamps in nanoseconds (because reasons), so we have to
        # quantize to microseconds
        t = self.events["t"] // 1e3; t -= t.min()
        self.events["t"] = t

        self.t_offset = 0
        self.t_final = t[-1]

        # the ms->idx mapping satisfies two conditions:
        # (1) t[ms_to_idx[ms]]     >= 1000*ms
        # (2) t[ms_to_idx[ms] - 1] <= 1000*ms
        # that is, the ms_to_idx maps the time in ms to the first event that occurred at least at that time.
        ms = np.arange(ceil(t[0] / 1e3), ceil(self.t_final / 1e3))
        self.ms_to_idx = np.searchsorted(t, 1e3 * ms)

    def get_final_time_us(self):
        return self.t_final

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        return super().get_events(t_start_us, t_end_us)


class EcdSequence(Sequence):
    def __init__(self, seq_path: Path, timebase: Timebase, num_bins=15, already_rectified=False):
        self.seq_name = PurePath(seq_path).name
        self.mode = 'test'
        self.name_idx = 0
        self.visualize_samples = False
        self.load_gt = False
        self.transforms = {}
        self.idx_to_visualize = []
        self.delta_t_us = timebase.dt

        self.num_bins = num_bins

        data_location = seq_path / "data.h5"
        self.h5f = h5py.File(data_location, "r")

        # ECD stores timestamps in nanoseconds (because reasons), so we have to
        # quantize to microseconds
        t: np.ndarray = np.array(self.h5f["events/t"]) // 1e3

        t_start = t[0] if timebase.start is None else max(t[0], timebase.start)
        t_end   = t[-1] if timebase.end is None else max(t[0], timebase.end)
        self.timestamps_flow = np.arange(start=self.delta_t_us, stop=t_end - t_start, step=self.delta_t_us)
        self.indices = np.arange(len(self.timestamps_flow))

        camera_info = self.h5f['camera_info/']
        self.width = camera_info.attrs['width']
        self.height = camera_info.attrs['height']
        self.camera_matrix = np.array(camera_info['K']).reshape(3, 3)
        self.distortion_parameters = np.array(camera_info['D'])

        self.voxel_grid = VoxelGrid((self.num_bins, self.height, self.width), normalize=True)
        self.event_slicer = EcdSlicer(self.h5f)

        if not already_rectified:
            pixel_points =(
                np.column_stack(np.unravel_index(np.arange(self.width * self.height), (self.height, self.width))[::-1])
                    .reshape(-1, 1, 2)
                    .astype(np.float32))

            self.rectify_ev_map =(
                cv2.undistortPoints(pixel_points, self.camera_matrix, self.distortion_parameters, P=self.camera_matrix)
                    .reshape(self.height, self.width, 2))

            self.already_rectified = False
        else:
            self.already_rectified = True

    def rectify_events(self, x: np.ndarray, y: np.ndarray, t: np.ndarray):
        # assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_map
        assert rectify_map.shape == (
            self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]


class MotionCompensatedEcdSequence(EcdSequence):
    def __init__(self, seq_path: Path, timebase: Timebase, num_bins=15):
        super().__init__(seq_path, timebase, num_bins, already_rectified=False)

        pixel_points = (
            np.column_stack(np.unravel_index(np.arange(self.width * self.height), (self.height, self.width))[::-1])
            .reshape(-1, 1, 2)
            .astype(np.float32))

        # our rectification map unprojects too
        self.rectify_ev_map = (
            cv2.undistortPoints(pixel_points, self.camera_matrix, self.distortion_parameters)
            .reshape(self.height, self.width, 2))

        self.camera_matrix = torch.from_numpy(self.camera_matrix).to(torch.float32).to('cuda')

        imu = self.h5f['imu']
        self.angular_velocity_camera_frame = {
            'x': np.array(imu['wx']), 'y': np.array(imu['wy']), 'z': np.array(imu['wz']),
            't': np.array(imu['t']) / 1e9 # ecd stores timestamps in nanoseconds...
        }

    def get_angular_velocity(self, ts: float) -> torch.tensor:
        """Returns the angular velocity at the provided microsecond timestamp in the world frame."""
        x_w = numpy.interp(ts / 1e6, self.angular_velocity_camera_frame['t'], self.angular_velocity_camera_frame['x'])
        y_w = numpy.interp(ts / 1e6, self.angular_velocity_camera_frame['t'], self.angular_velocity_camera_frame['y'])
        z_w = numpy.interp(ts / 1e6, self.angular_velocity_camera_frame['t'], self.angular_velocity_camera_frame['z'])
        return torch.tensor([ x_w, y_w, z_w ])

    def euler_to_rotation_matrices(self, euler_angles: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of Euler angles (ZYX / yaw-pitch-roll convention) to rotation matrices.
        args:
            euler_angles: (N, 3) tensor of [roll, pitch, yaw] in radians.
        Returns:
            (N, 3, 3) tensor of rotation matrices.
        """
        roll  = euler_angles[:, 0].to('cuda')  # Rotation about X
        pitch = euler_angles[:, 1].to('cuda')  # Rotation about Y
        yaw   = euler_angles[:, 2].to('cuda')  # Rotation about Z

        cos_r, sin_r = torch.cos(roll), torch.sin(roll)
        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

        zeros = torch.zeros_like(roll)
        ones = torch.ones_like(roll)

        Rx = torch.stack([
            ones, zeros, zeros,
            zeros, cos_r, -sin_r,
            zeros, sin_r, cos_r,
        ], dim=-1).reshape(-1, 3, 3)

        Ry = torch.stack([
            cos_p, zeros, sin_p,
            zeros, ones, zeros,
            -sin_p, zeros, cos_p,
        ], dim=-1).reshape(-1, 3, 3)

        Rz = torch.stack([
            cos_y, -sin_y, zeros,
            sin_y, cos_y, zeros,
            zeros, zeros, ones,
        ], dim=-1).reshape(-1, 3, 3)

        # Combined rotation: R = Rz @ Ry @ Rx  (intrinsic X-Y-Z = extrinsic Z-Y-X)
        return (Rz @ Ry @ Rx).to(torch.float32)

    def rectify_events(self, x: np.ndarray, y: np.ndarray, t: np.ndarray):
        # unproject and normalize
        unprojected = np.column_stack((super(EcdSequence, self).rectify_events(x, y, t), np.ones_like(x)))
        unprojected /= np.linalg.norm(unprojected, axis=-1)[:, None]
        unprojected = torch.from_numpy(unprojected).to('cuda')

        # compensate, assuming constant angular velocity
        t_init = float(t[0]); t_final = float(t[-1])
        w_approx = self.get_angular_velocity((t_final + t_init) / 2)
        angle = w_approx[None, :] * ((t[:, None] - t_init) / 1e6)

        R = self.euler_to_rotation_matrices(angle)
        compensated = torch.bmm(R, unprojected.unsqueeze(-1)).squeeze(-1)

        reprojected = (self.camera_matrix @ compensated.T).T
        reprojected = reprojected[:, 0:2] / reprojected[:, 2, None]

        return reprojected.cpu().detach().numpy()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='id-4x', choices=['id-4x', 'id-8x', 'tid'], help='name of the model to use')
    parser.add_argument("--sequences", type=str, default="", help="comma-separated list of sequences to inference")
    parser.add_argument("--cuda-device", type=str, default='cuda:0', help="torch-style CUDA device identifier")
    parser.add_argument("--dataset-type", type=str, default='dsec', choices=['dsec', 'ecd'], help="format of the data to load")

    parser.add_argument("--compensate", action='store_true', help="indicates the events should be motion-compensated using the dataset's angular velocity")
    parser.add_argument("--delta-time", "-dt", type=int, default=100_000, help="the time (in us) between flow estimates")
    parser.add_argument("--start-time", '-st', type=int, default=None, help="the time (in us) of the first flow estimate")
    parser.add_argument("--end-time", '-et', type=int, default=None, help="the maximum time (in us) of the last flow estimate")

    parser.add_argument("data_dir", type=str, help='directory in which dataset is stored')
    parser.add_argument("results_dir", type=str, help='directory in which to store results')
    return parser.parse_args()

def create_model(model_name: str) -> Tuple[IDEDEQIDO, str, str]:
    model_type = MODEL_TYPE[model_name]
    model_config = types.SimpleNamespace(**(MODEL_CONFIG[model_type] | MODEL_CONFIG_OVERRIDES[model_name]))
    model = get_model_by_name(model_type, model_config)

    ckpt = torch.load(model_config.pretrain_ckpt, map_location='cuda:0')
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        try:
            model.load_state_dict(ckpt)
        except:
            raise ValueError("Invalid checkpoint")

    return model, model_name, model_type

def forward(model: IDEDEQIDO, batch):
    """
    Stand-in for the logic on lines 95, 96 of execute_test() in idn.tests.dsec.
    """
    if isinstance(model, RecIDE):
        return model.forward(batch[0])
    else:
        return model.forward(batch)

@contextmanager
def evaluate_model(self, model, cuda_device: str):
    original_device = next(model.parameters()).device
    try:
        model.cuda(cuda_device)
        if isinstance(model, RecIDE):
            model.co_mode = True
            model.reset_continuous_flow()
        model.eval()
        yield model
    finally:
        self.cleanup_model(model)
        model.to(original_device)

def load_data(dataset_type: str, dataset_config: Dict, data_root: Path, sequences: List[str], timebase: Timebase, compensate=False) -> List[DataLoader]:
    if len(sequences) == 0: sequences = list(data_root.iterdir())
    else: sequences = [ data_root / sequence for sequence in sequences ]

    if dataset_type == 'dsec':
        sequences = assemble_dsec_test_set(
            data_root,
            seq_len=dataset_config.get("sequence_length", None), representation_type=RepresentationType.VOXEL,
        )
    elif dataset_type == 'ecd':
        # todo: need to support "recurrent" sequences for TID.
        sequence_cls = MotionCompensatedEcdSequence if compensate else EcdSequence
        sequences = [sequence_cls(it, timebase) for it in filter(lambda sequence: (sequence / "data.h5").is_file(), sequences)]

    collate_fn = rec_train_collate if dataset_config.get("recurrent", False) else train_collate
    return [ DataLoader(seq, collate_fn=collate_fn, **DATALOADER_CONFIG) for seq in sequences ]

def main():
    args = parse_arguments()

    # ensure data root exists
    data_root = Path(args.data_dir)
    if not data_root.is_dir(): raise ValueError(f"data directory '{args.data_dir}' does not exist.")

    # load model
    model, model_name, model_type = create_model(args.model)
    print(f"Loaded model '{model_name}'.")

    # load data
    sequences = args.sequences.split(',')
    dataset_config = DATASET_CONFIG | DATASET_CONFIG_OVERRIDES[model_type]
    data_loader = load_data(
        args.dataset_type, dataset_config,
        data_root, sequences,
        compensate=args.compensate,
        timebase=Timebase(start=args.start_time, end=args.end_time, dt=args.delta_time)
    )

    # configure model
    model.cuda(args.cuda_device)
    if isinstance(model, RecIDE):
        model.co_mode = True
        model.reset_continuous_flow()
    model.eval()

    # inference each sequence
    with torch.no_grad():
        for sequence in data_loader:
            sequence_name = sequence.dataset.seq_name

            flows = [None] * len(sequence)
            for idx, batch in enumerate(tqdm(sequence, position=0, leave=True, desc=f"Sequence '{sequence_name}'")):
                batch = move_batch_to_cuda(batch, args.cuda_device)
                out = forward(model, batch)
                flows[idx] = out['final_prediction'].cpu().numpy().squeeze()

            path = Path(f"{args.results_dir}/{dataset_config['dataset_name']}/{sequence_name}/flow")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, flows=np.stack(flows).astype(np.float16), timestamps=sequence.dataset.timestamps_flow.astype(np.uint32))

if __name__ == '__main__':
    main()