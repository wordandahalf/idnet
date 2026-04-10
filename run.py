import argparse
import types
from contextlib import contextmanager
from math import ceil
from pathlib import Path, PurePath
from typing import Tuple, List, Dict

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
    def __init__(self, seq_path: Path, num_bins=15, delta_t_ms=100):
        self.seq_name = PurePath(seq_path).name
        self.mode = 'test'
        self.name_idx = 0
        self.visualize_samples = False
        self.load_gt = False
        self.transforms = {}
        self.idx_to_visualize = []
        self.delta_t_us = delta_t_ms * 1000

        self.num_bins = num_bins

        data_location = seq_path / "data.h5"
        self.h5f = h5py.File(data_location, "r")

        # ECD stores timestamps in nanoseconds (because reasons), so we have to
        # quantize to microseconds
        t = self.h5f["events/t"]
        self.timestamps_flow = np.arange(start=self.delta_t_us, stop=(t[-1] - t[0]) // 1e3, step=self.delta_t_us)
        self.indices = np.arange(len(self.timestamps_flow))

        camera_info = self.h5f['camera_info/']
        self.width = camera_info.attrs['width']
        self.height = camera_info.attrs['height']
        self.camera_matrix = np.array(camera_info['K']).reshape(3, 3)
        self.distortion_parameters = np.array(camera_info['D'])

        self.voxel_grid = VoxelGrid((self.num_bins, self.height, self.width), normalize=True)
        self.event_slicer = EcdSlicer(self.h5f)

        pixel_points =(
            np.column_stack(np.unravel_index(np.arange(self.width*self.height), (self.height, self.width)))
                .reshape(-1, 1, 2)
                .astype(np.float32))
        self.rectify_ev_map =(
            cv2.undistortPoints(pixel_points, self.camera_matrix, self.distortion_parameters, P=self.camera_matrix)
                .reshape(self.height, self.width, 2))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='id-4x', choices=['id-4x', 'id-8x', 'tid'], help='name of the model to use')
    parser.add_argument("--sequences", type=str, default="", help="comma-separated list of sequences to inference")
    parser.add_argument("--cuda-device", type=str, default='cuda:0', help="torch-style CUDA device identifier")
    parser.add_argument("--dataset-type", type=str, default='dsec', choices=['dsec', 'ecd'], help="format of the data to load")
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

def load_data(dataset_type: str, dataset_config: Dict, data_root: Path, sequences: List[str]) -> List[DataLoader]:
    if len(sequences) == 0: sequences = list(data_root.iterdir())
    else: sequences = [ data_root / sequence for sequence in sequences ]

    if dataset_type == 'dsec':
        sequences = assemble_dsec_test_set(
            data_root,
            seq_len=dataset_config.get("sequence_length", None), representation_type=RepresentationType.VOXEL,
        )
    elif dataset_type == 'ecd':
        # todo: need to support "recurrent" sequences for TID.
        sequences = [EcdSequence(it) for it in filter(lambda sequence: (sequence / "data.h5").is_file(), sequences)]

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
    data_loader = load_data(args.dataset_type, dataset_config, data_root, sequences)

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
            np.save(path, np.stack(flows))

if __name__ == '__main__':
    main()