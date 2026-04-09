# Lightweight Event-based Optical Flow Estimation via Iterative Deblurring

Work accepted to 2024 IEEE International Conference on Robotics and Automation (ICRA'24) [[paper](https://arxiv.org/abs/2211.13726), [video](https://www.youtube.com/watch?v=1qA1hONS4Sw)].


<img width="1024" alt="idnet-graphical-abstract" src="https://github.com/tudelft/idnet/assets/10345786/0fa638b2-583e-4ad4-99b1-7bcc9a5e465f">


![id-viz](https://github.com/tudelft/idnet/assets/10345786/f6314f3a-7e24-444a-bd28-695267ede7b4)

If you use this code in an academic context, please cite our work:

```bibtex
@INPROCEEDINGS{10610353,
  author={Wu, Yilun and Paredes-Vallés, Federico and de Croon, Guido C. H. E.},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Lightweight Event-based Optical Flow Estimation via Iterative Deblurring}, 
  year={2024},
  volume={},
  number={},
  pages={14708-14715},
  keywords={Image motion analysis;Correlation;Memory management;Estimation;Rendering (computer graphics);Real-time systems;Iterative algorithms},
  doi={10.1109/ICRA57147.2024.10610353}}
```

## Dependencies
Create a virtual env and install dependencies by running:
```
pip -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
The code is tested working with Python 3.13.

## Download (For Evaluation)
The DSEC dataset for optical flow can be downloaded [here](https://dsec.ifi.uzh.ch/dsec-datasets/download/).
Use script [download_dsec_test.py](scripts/download-dsec-test.py) for your convenience.
It downloads the dataset directly into the `DATA_DIRECTORY` with the expected directory structure.
```python
scripts/download-dsec-test.py <DATA_DIRECTORY>
```
Once downloaded, create a symbolic link called  `data` pointing to the data directory:
```
ln -s <DATA_DIRECTORY> data/test
```

## Download (For Training)
For training on DSEC, two more folders need to be downloaded:

- Unzip [train_events.zip](https://download.ifi.uzh.ch/rpg/DSEC/train_coarse/train_events.zip) to data/train_events

- Unzip [train_optical_flow.zip](https://download.ifi.uzh.ch/rpg/DSEC/train_coarse/train_optical_flow.zip) to data/train_optical_flow

or establish symbolic links under data/ pointing to the folders.

## Download (MVSEC)
To run experiments on MVSEC, additionally download outdoor day sequences .h5 files from https://drive.google.com/open?id=1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV
and place the files under data/ or point symbolic links pointing to the data files under data/.

## Run Evaluation

To run eval:
```
cd idnet
conda activate IDNet
python -m idn.eval
```

Change the save directory for eval results in `idn/config/validation/dsec_test.yaml` if you prefer. The default is at `/tmp/collect/XX`.

To switch between models, change the model option in `idn/config/id_eval.yaml` to switch between id model with 1/4 and 1/8 resolution.

To eval TID model, change the function decorator above the main function in `eval.py`.

At the end of evaluation, a zip file containing the results will be created in the save directory, for which you can upload to the DSEC benchmark website to reproduce our results.

## Run Training
To train IDNet, run:
```
cd idnet
conda activate IDNet
python -m idn.train
```

Similarly, switch between id-4x, id-8x and tid models and MVSEC training by changing the hydra.main() decorator in `train.py` and settings in the corresponding .yaml file.

