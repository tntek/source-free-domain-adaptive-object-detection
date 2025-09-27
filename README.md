## Installation Instructions
- We use Python 3.8, PyTorch 1.13.1 (CUDA 11.7 build).
- The codebase is built on [Detectron](https://github.com/facebookresearch/detectron2).

```angular2
conda create -n wsco python=3.8

Conda activate wsco

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

cd wsco
pip install -r requirements.txt

## Make sure you have GCC and G++ version <=8.0
cd ..
python -m pip install -e wsco

```
## detectron2
python setup.py build develop


## Dataset Preparation

* **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets.
* **Clipart, WaterColor**: Dataset preparation instruction link [Cross Domain Detection ](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets). Images translated by Cyclegan are available in the website.
* **Sim10k**: Website [Sim10k](https://fcav.engin.umich.edu/sim-dataset/)
* **CitysScape, FoggyCityscape**: Download website [Cityscape](https://www.cityscapes-dataset.com/), see dataset preparation code in [DA-Faster RCNN](https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch)

Download all the dataset into "./dataset" folder.
The codes are written to fit for the format of PASCAL_VOC.
For example, the dataset [Sim10k](https://fcav.engin.umich.edu/sim-dataset/) is stored as follows.

```
$ cd ./dataset/Sim10k/VOC2012/
$ ls
Annotations  ImageSets  JPEGImages
$ cat ImageSets/Main/val.txt
3384827.jpg
3384828.jpg
3384829.jpg
.
.
```

## Execution Instructions

### Training

- Please download the source training model and style training model (from LODS) and put them in the WSCo directory. [link](https://drive.google.com/drive/folders/1xxzfnzZfjJiTEklVxgUDCVrS6wxG47bd?usp=sharing)

for example :
WSCO
  source_model/
  style_models/

## train model

sh train.sh



### Evaluation

- After training, load the teacher model weights and perform evaluation using
```angular2
python tools/plain_test_net.py --model-dir "/home/eason/AAAI/aa/checkpoint/foggy/2024-12-29-01:46:39_lpld/model_teacher_$t.pth" --where lpld --config-file analysis/sfda_foggy.yaml

```
