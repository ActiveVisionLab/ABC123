# Learning To Count Anything: Reference-less Class-agnostic Counting with Weak Supervision
**[Project Page](https://ABC123.active.vision/) |
[Latest arXiv](https://arxiv.org/abs/2309.04820) |
[Dataset](https://MCAC.active.vision)
**

[Michael Hobley](https://scholar.google.co.uk/citations?user=2EftbyIAAAAJ&hl=en), 
[Victor Adrian Prisacariu](http://www.robots.ox.ac.uk/~victor/). 

[Active Vision Lab (AVL)](https://www.robots.ox.ac.uk/~lav/),
University of Oxford.


## Environment
We provide a `environment.yml` file to set up a `conda` environment:

```sh
git clone https://github.com/ActiveVisionLab/ABC123.git
cd ABC123
conda env create -f environment.yml
```

## Dataset Download 
### MCAC
Dowload [MCAC](https://www.robots.ox.ac.uk/~lav/Datasets/MCAC/MCAC.zip)
to precompute ground truth density maps for other resolutions, occlusion percentages, and gaussian standard deviations:

```sh
cd PATH/TO/MCAC/
python make_gaussian_maps.py  --occulsion_limit <desired_max_occlusion>  --crop_size 672 --img_size <desired_resolution> --gauss_constant <desired_gaussian_std>;
```

## Trained Model Download 
We provide [example weights](https://www.robots.ox.ac.uk/~mahobley/ABC123/model_chkpt.zip) for our models trained on MCAC.
Put this in ./checkpoints/.


## Example Training 

To train the counting network:
```sh
python main.py --config ABC123;
```

## Example Testing
To test a trained model on MCAC: 

```sh
python main.py --config ABC123test;
```

To test a trained model on MCAC-M1: 

```sh
python main.py --config ABC123testM1;
```



## Citation
```
@article{hobley2023abc,
    title={ABC Easy as 123: A Blind Counter for Exemplar-Free Multi-Class Class-agnostic Counting}, 
    author={Michael A. Hobley and Victor A. Prisacariu},
    journal={arXiv preprint arXiv:2309.04820},
    year={2023},
}
```