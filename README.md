# Privacy-AdversarialLearning
## TensorFlow Code for 'Towards Privacy-Preserving Visual Recognition via Adversarial Training: A Pilot Study'

## Introduction

TensorFlow Implementation of our ECCV 2018 paper ["Towards Privacy-Preserving Visual Recognition via Adversarial Training: A Pilot Study"](https://arxiv.org/abs/1807.08379) and our TPAMI (under review) paper ["Privacy-Preserving Deep Visual Recognition: An Adversarial Learning Framework and A New Dataset"](https://arxiv.org/pdf/1906.05675.pdf).

This paper aims to improve privacy-preserving visual recognition, an increasingly demanded feature in smart camera applications, by formulating a unique adversarial training framework.

The proposed framework explicitly learns a degradation transform for the original video inputs, in order to optimize the trade-off between target task performance and the associated privacy budgets on the degraded video. A notable challenge is that the privacy budget, often defined and measured in task-driven contexts, cannot be reliably indicated using any single model performance, because a strong protection of privacy has to sustain against any possible model that tries to hack privacy information.

Such an uncommon situation has motivated us to propose two strategies to enhance the generalization of the learned degradation on protecting privacy against unseen hacker models. Novel training strategies, evaluation protocols, and result visualization methods have been designed accordingly.

Two experiments on privacy-preserving action recognition, with privacy budgets defined in various ways, manifest the compelling effectiveness of the proposed framework in simultaneously maintaining high target task (action recognition) performance while suppressing the privacy breach risk.
## Pretrained checkpoints for adversarial training on SBU
Google Drive: https://drive.google.com/file/d/1dIwr7JrFVkuo9X0WXpSqkzQNr_jhRzv7/view?usp=sharing

## OpenCV with FFMPEG support

### Install dependencies
```
sudo apt install gcc g++ git libjpeg-dev libpng-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev pkg-config cmake libgtk2.0-dev libeigen3-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev sphinx-common libtbb-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libavcodec-dev libavutil-dev libavfilter-dev libavformat-dev libavresample-dev
```
### Download OpenCV
```
wget https://github.com/Itseez/opencv/archive/3.4.0.zip
unzip 3.4.0.zip
cd opencv-3.4.0
mkdir build
cd build
```
### Installing OpenCV with cmake
```
cmake -D WITH_FFMPEG=ON -D WITH_CUDA=OFF -D BUILD_TIFF=ON -D BUILD_opencv_java=OFF -D ENABLE_AVX=ON -D WITH_OPENGL=ON -D WITH_OPENCL=ON -D WITH_IPP=ON -D WITH_TBB=ON -D WITH_EIGEN=ON -D WITH_V4L=ON -D WITH_VTK=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_opencv_python2=OFF -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") -D PYTHON3_EXECUTABLE=$(which python3) -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=OFF -D PYTHON_EXECUTABLE=/home/wuzhenyu_sjtu/anaconda2/envs/tf-1.8-py36/bin/python -D BUILD_EXAMPLES=ON ..
make -j 128
sudo make install
```
### Check the success of installing OpenCV with FFMPEG support
```
# If OpenCV has FFMPEG support, it will give "FFMPEG: YES"
python -c "import cv2; print(cv2.getBuildInformation())" | grep -i ffmpeg
```
## Datasets
SBU: https://www3.cs.stonybrook.edu/~kyun/research/kinect_interaction/index.html (Please download the clean version)

HMDB51: http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar

UCF101: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

VISPR: https://tribhuvanesh.github.io/vpa/

PA-HMDB51: https://github.com/TAMU-VITA/PA-HMDB51

## Models for two-fold evaluation
We have used Inception-v1, Inception-V2, MobileNet-V1, ResNet-V1 and ResNet-V2. They are available at https://github.com/tensorflow/models/tree/master/research/slim/nets.

## Privacy Preserving in Smart Home

### File path
* SBU
  * data (please copy to this path)
  * pretrain: https://drive.google.com/drive/u/0/folders/1bD-e4vtMPF04sW3vEBIWVLaPDlLaeYKl
    * C3D (please copy to this path)
    * target_models
    * degradation_models (please copy to this path)
    * budget_k%d
  * adversarial_training
  * two_fold_evaluation
    * pretrained_budget_model (please copy to this path)

### Commands
Train and test using kbeam: (run this command under adversarial_training dir)
```
python main_kbeam.py --GPU_id 0,1 --_K 4
```
Two-fold validate on kbeam model: (run this command under two_fold_evaluation dir)
```
python main_2fold.py --GPU_id 0,1,2,3 --adversarial_job_name kbeam-NoRest-K4
```

## Dependencies

Python 3.5
* [TensorFlow 1.8.0](https://www.tensorflow.org/)

## Citation

If you find this code useful, please cite the following paper:
```BibTex
@inproceedings{wu2018towards,
  title={Towards privacy-preserving visual recognition via adversarial training: A pilot study},
  author={Wu, Zhenyu and Wang, Zhangyang and Wang, Zhaowen and Jin, Hailin},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={606--624},
  year={2018}
}
```
```BibTex
@article{wang2019privacy,
  title={Privacy-Preserving Deep Visual Recognition: An Adversarial Learning Framework and A New Dataset},
  author={Wang, Haotao and Wu, Zhenyu and Wang, Zhangyang and Wang, Zhaowen and Jin, Hailin},
  journal={arXiv preprint arXiv:1906.05675},
  year={2019}
}
```
