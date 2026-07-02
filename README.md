# EAR-Net

This repository provideds the code associated wth our paper: **EAR-Net: Pursuing End-to-End Absolute Rotations from Multi-View Images**

## Dependencies

- pytorch == 1.13.1

## Usage

### Dataset

Download and prepare [ScanNet](http://www.scan-net.org/) according to the official document.

Then create symlinks from the downloaded dataset to `./data/ScanNet`

### Train

The training involves two stages, pretraining and end-to-end training.

- Pretrain

During the pretraining stage, the encoder and the rotation branch is trained:

```sh
bash dist_train.sh --configs configs/pretrain.yaml
```

- End-to-end training

During the end-to-end training stage, the encoder, the rotation branch and the confidence branch are jointly trained:

```sh
bash dist_train.sh --configs configs/end-to-end.yaml
```

### Test

```sh
python test.py --configs configs/end-to-end.yaml
```


## Citation

```
@article{liu2026earnet,
  title={EAR-Net: Pursuing End-to-End Absolute Rotations from Multi-View Images},
  author={Liu, Y. and Dong, Q.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
}
```
