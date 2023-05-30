# Deep Clustering with Incomplete Noisy Pairwise Annotations: A Geometric Regularization Approach

This repo provides Python implementation of the work "Deep Clustering with Incomplete Noisy Pairwise Annotations: A Geometric Regularization Approach", accepted to ICML 2023.


## Setup

- Clone this project: `git clone https://github.com/ductri/VolMaxDCC`
- `cd VolMaxDCC`
- (Optional but strongly suggest) Create a separate virtual environment using python3: `python3 -m venv localenv`, then activate it `source localenv/bin/activate`
- Install all package listed in requirements.txt: `pip install -r requirements.txt`



## Training
We have set a default configution. Without any change, you should able to perform:

- Training the experiment on ImageNet10 in noiseless pairwise setting:
```
python src/our_model__train_demo_imagenet10.py
```
This will load a pairwise labels dataset that have been drawn randomly and stored to datasets/. This dataset containing 10k pairs drawn randomly from the training part of ImageNet10.

- Evaluate performance in terms of ACC, NMI, and ARI:
```
python src/our_model__eval.py
```
This will evaluate the learned mapping f using 2k test dataset.


Other options:
- You can create different pairwise dataset by inspecting file `imagenet10_create_pair.py`, similarly for stl10 and cifar10.

## Datasets

### Real pairwise annotations
- Create ImageNet-10 from ImageNet with the following classes: 
```
n02056570
n02085936
n02128757
n02690373
n02692877
n03095699
n04254680
n04285008
n04467665
n07747607
```
Download all images of these class and store them in some directory, for example `imagenet10/raw`. The folder should look like
```
imagenet10/raw
           |---n02056570/
           |---n02085936/
           |---n02128757/
           |---n02690373/
           |---n02692877/
           |---n03095699/
           |---n04254680/
           |---n04285008/
           |---n04467665/
           |---n07747607/
```
- Load the dataset using
```
dataset = torchvision.datasets.ImageFolder('imagenet10/raw')
```
The order of this `dataset` __matters__.

- Load pairwise indices
```
import pickle as pkl
with open('datasets/imagenet10/pairs/pair_8994_real_imagenet10-cc-10k_trial_0.pkl', 'rb') as i_f:
    pairwise_data = pkl.load(i_f)
```

`pairwise_data` is a `dict` with following keys:

- `shuffle_inds`: a permutation of `range(13000)`. This permutation is used to shuffle order of the `dataset`.
- `ind_pairs`: list of 8994 pairs of indices with respect to `shuffle_inds`.
- `label_pairs`: a list of 8994 pairwise labels of `ind_pairs`, annotated by AMT workers.
- `true_label_pairs`: a list of 8994 pairwise labels of `ind_pairs`, but inferred from ground truth class label.
- `X`: feature vector extracted from a pretrained unsupervised method.
- `true_y`: corresponding class label of the data `X`.

