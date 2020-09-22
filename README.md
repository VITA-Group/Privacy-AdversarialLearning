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

# PA-HMDB51
This is the repo for PA-HMDB51 (privacy attribute HMDB51) dataset published in our paper http://arxiv.org/abs/1906.05675.

This dataset is collected and maintained by the [VITA group](https://www.atlaswang.com/group) at the CSE department of Texas A&M University.


## Overview
PA-HMDB51 is the very first human action video dataset with both privacy attributes and action labels provided. The dataset contains 592 videos selected from HMDB51 [1], each provided with frame-level annotation of five privacy attributes. We evaluated the visual privacy algorithms proposed in [3] on PA-HMDB51.

## Privacy attributes
We carefully selected five privacy attributes, which are originally from the 68 privacy attributes defined in [2], to annotate. The definition of the five attributes can be found in the following table.

<!-- ![PA def table](https://github.com/htwang14/PA-HMDB51/blob/master/imgs/def_table.PNG)-->

<table id="Main table">
    <thead>
        <tr>
            <th>Attribute</th>
            <th>Possible Values</th>
            <th>Meaning</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4> Skin Color </td>
            <td> 0 invisible </td>
            <td> Skin color of the actor is invisible. </td>
        </tr>
        <tr>
            <td> 1 white </td>
            <td> Skin color of the actor is white. </td>
        </tr>
        <tr>
            <td> 2 brown/yellow </td>
            <td> Skin color of the actor is brown/yellow. </td>
        </tr>
        <tr>
            <td> 3 black </td>
            <td> Skin color of the actor is black. </td>
        </tr>
        <tr>
            <td rowspan=3> Face </td>
            <td> 0 No face </td>
            <td> Less than 10% of the actor’s face is visible.  </td>
        </tr>
        <tr>
            <td> 1 Partial face </td>
            <td> Less than 70% but more than 10% of the actor’s face is visible. </td>
        </tr>
        <tr>
            <td> 2 Whole face </td>
            <td> More than 70% of the actor’s face is visible. </td>
        </tr>
        <tr>
            <td rowspan=3> Gender </td>
            <td> 0 Cannot tell </td>
            <td> Cannot tell the person’s gender.  </td>
        </tr>
        <tr>
            <td> 1 Male </td>
            <td> It’s an actor. </td>
        </tr>
        <tr>
            <td> 2 Female </td>
            <td> It’s an actress. </td>
        </tr>
        <tr>
            <td rowspan=3> Nudity </td>
            <td> 0 </td>
            <td> The actor/actress is wearing long sleeves and pants.  </td>
        </tr>
        <tr>
            <td> 1 </td>
            <td> The actor/actress is wearing short sleeves or shorts/short skirts. </td>
        </tr>
        <tr>
            <td> 2 </td>
            <td> The actor/actress is of semi-nudity. </td>
        </tr>
        <tr>
            <td rowspan=2> Relationship </td>
            <td> 0 Cannot tell </td>
            <td> Relationships (such as friends, couples, etc.) between the actors/actress cannot be told from the video.   </td>
        </tr>
        <tr>
            <td> 1 Can tell </td>
            <td> Relationships between the actors/actress can be told from the video. </td>
        </tr>
    </tbody>
</table>


## Examples
| Frame             |  Action | Privacy Attributes |
|:-------------------------:|:-------------------------:|:----------------------:|
| ![](https://github.com/htwang14/PA-HMDB51/blob/master/imgs/brush_hair.png) | brush hair | skin color: white <br> face: no <br> gender: female <br> nudity: level 2 <br> relationship: no |
| ![](https://github.com/htwang14/PA-HMDB51/blob/master/imgs/pullup.png) | pullup | skin color: white <br> face: whole <br> gender: male <br> nudity: level 1 <br> relationship: no |

## Download link
[Google drive](https://drive.google.com/drive/u/0/folders/1OtQLtq9QxdPHaH1gUcFZiylBMXJhn2dm)

## Label format
The attributes usually don't change that much across a video, so we only need to label the starting and ending frame index of each attribute.
For example, if a video has 100 frames, and we can see a complete human face in the first 50 frames while a partial face in the next 50 frames, we would label [face: complete, s: 0, e: 49], [face: partial, s: 50, e: 99], where 's' is for 'starting' frame and 'e' is for 'ending' frame.
Note that each attribute is labeled separately.
For instance, if the actor's skin color is visible in all 100 frames in the same video (assume the actor is white), we will label [skin color: white, s: 0, e: 99].
The privacy attributes for all 'brush hair' videos are in brush_hair.json, similar with all other actions.


## Acknowledgements
We sincerely thank Scott Hoang, James Ault, Prateek Shroff, [Zhenyu Wu](https://wuzhenyusjtu.github.io/) and [Haotao Wang](http://people.tamu.edu/~htwang/) for labeling the dataset.


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
@article{wang2019privacy,
  title={Privacy-Preserving Deep Visual Recognition: An Adversarial Learning Framework and A New Dataset},
  author={Wang, Haotao and Wu, Zhenyu and Wang, Zhangyang and Wang, Zhaowen and Jin, Hailin},
  journal={arXiv preprint arXiv:1906.05675},
  year={2019}
}
```
