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
@ARTICLE{11586175,
  author={Liu, Yuzhen and Dong, Qiulei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={EAR-Net: Pursuing End-to-End Absolute Rotations From Multi-View Images}, 
  year={2026},
  volume={48},
  number={10},
  pages={13533-13548},
  doi={10.1109/TPAMI.2026.3708244}
}
```
