import argparse
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple
import os
import imageio

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from idn.loader.loader_dsec import rec_train_collate, train_collate, assemble_dsec_test_set
from idn.model.idedeq import IDEDEQIDO, RecIDE
from idn.utils.dsec_utils import RepresentationType
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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='id-4x', choices=['id-4x', 'id-8x', 'tid'], help='name of the model to use')
    parser.add_argument("--sequences", type=str, default="", help="comma-separated list of sequences to inference")
    parser.add_argument("--cuda-device", type=str, default='cuda:0', help="torch-style CUDA device identifier")
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

def save_submission(path, flow, file_idx):
    os.makedirs(os.path.join(path, "submission"), exist_ok=True)
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()
    assert flow.shape == (1, 2, 480, 640)
    flow = flow.squeeze()
    _, h, w = flow.shape
    scaled_flow = np.rint(
        flow*128 + 2**15).astype(np.uint16).transpose(1, 2, 0)
    flow_image = np.concatenate((scaled_flow, np.zeros((h, w, 1),
                                dtype=np.uint16)), axis=-1)
    imageio.imwrite(os.path.join(path, "submission", f"{file_idx:06d}.png"), flow_image, format='PNG-FI')

def main():
    args = parse_arguments()

    model, model_name, model_type = create_model(args.model)
    print(f"Loaded model '{model_name}'.")

    # load data sequences
    dataset_config = DATASET_CONFIG | DATASET_CONFIG_OVERRIDES[model_type]
    valid_set = assemble_dsec_test_set(
        args.data_dir,
        seq_len=dataset_config.get("sequence_length", None), representation_type=RepresentationType.VOXEL,
    )
    collate_fn = rec_train_collate if dataset_config.get("recurrent", False) else train_collate
    if isinstance(valid_set, list):
        data_loader = [DataLoader(
            seq, collate_fn=collate_fn, **DATALOADER_CONFIG
        ) for seq in valid_set]
    else:
        data_loader = [DataLoader(
            valid_set, collate_fn=collate_fn, **DATALOADER_CONFIG
        )]

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