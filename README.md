# SSOLE

Implementation for paper [SSOLE: Rethinking Orthogonal Low-rank Embedding for Self-Supervised Learning](https://openreview.net/forum?id=zBgiCWCxJB) (ICLR 2025)

## Requirements
```
PyTorch
PyTorchLightning
```

## Training

```bash
python train.py --config configs/ssole_imagenet100_m4.yaml
```

## Configuration

Edit the configuration files in the `configs/` directory to customize training parameters.

## Citation
```
@inproceedings{huang2025ssole,
  title={SSOLE: Rethinking Orthogonal Low-rank Embedding for Self-Supervised Learning},
  author={Huang, Lun and Qiu, Qiang and Sapiro, Guillermo},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
