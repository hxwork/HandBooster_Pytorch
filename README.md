# [CVPR 2024] HandBooster: Boosting 3D Hand-Mesh Reconstruction by Conditional Synthesis and Sampling of Hand-Object Interactions

<h4 align = "center">Hao Xu<sup>1</sup>, Haipeng Li<sup>2</sup>, Yinqiao Wang<sup>1</sup>, Shuaicheng Liu<sup>2</sup>, Chi-Wing Fu<sup>1</sup></h4>
<!-- <h4 align = "center"> <sup>1</sup>Department of Computer Science and Engineering,     <sup>2</sup>Institute of Medical Intelligence and XR</center></h4> -->
<h4 align = "center"> <sup>1</sup>The Chinese University of Hong Kong</center></h4>
<h4 align = "center"> <sup>2</sup>University of Electronic Science and Technology of China</center></h4>

This is the official implementation of our CVPR2024 paper, [HandBooster: Boosting 3D Hand-Mesh Reconstruction by Conditional Synthesis and Sampling of Hand-Object Interactions](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_HandBooster_Boosting_3D_Hand-Mesh_Reconstruction_by_Conditional_Synthesis_and_Sampling_CVPR_2024_paper.pdf).

Here is our presentation video: [YouTube](https://www.youtube.com/watch?v=Eg_LfnFef9g)

## Our Poster

![poster](./assets/poster.png)

## Todo List

We release following key parts, but there are still some issues need to be fixed caused by transfering code from server to local environment.

* [X] Novel condition creator
* [X] Hand-object image synthesis
* [X] Boosted hand-mesh reconstruction baselines
* [ ] Fix small issues

## Install

* Environment
  ```
  conda create -n handbooster python=3.8
  conda activate handbooster
  ```
* Requirements
  ```
  pip install -r requirements.txt
  ```

## Data Preparation

We evaluate different models on the DexYCB and HO3D datasets. The pre-processed ground truths are from [HandOccNet](https://github.com/namepllet/HandOccNet). Please follow its instruction to prepare the data and ground truths like this,

```
|-- data  
|   |-- HO3D
|   |   |-- train
|   |   |   |-- ABF10
|   |   |   |-- ......
|   |   |-- evaluation
|   |   |-- annotations
|   |   |   |-- HO3D_train_data.json
|   |   |   |-- HO3D_evaluation_data.json
|   |-- DEX_YCB
|   |   |-- 20200709-subject-01
|   |   |-- ......
|   |   |-- annotations
|   |   |   |--DEX_YCB_s0_train_data.json
|   |   |   |--DEX_YCB_s0_test_data.json

```

Also, you need to download the MANO models (MANO_LEFT.pkl and MANO_RIGHT.pkl) from its official website and put it into `NovelConditionCreator/DexGraspNet/grasp_generation/mano, NovelConditionCreator/dexycb/manopth/mano/models, NovelConditionCreator/ho3d/manopth/mano/models, HandObjectImageSynthesizer/hand_recon/common/utils/manopth/mano/models, HandReconstruction/common/utils/manopth/mano/models`. You can change the model path if you don't want to copy them several times.

## Novel Condition Creator

#### Step 1: Pose simulation and validation

We follow the DexGraspNet to similuate grasping poses. We remove the table penalty. Check `NovelConditionCreator/DexGraspNet/grasp_generation/main.py` for the pose generation and `NovelConditionCreator/DexGraspNet/grasp_generation/pose_validation.py` for the pose validation.

#### Step2: Condition preparation

We preprocess the simulated grasping poses to the required image condition in the next step. Check ` NovelConditionCreator/dexycb/dexycb_preprocess.py` and  `NovelConditionCreator/ho3d/ho3d_preprocess.py` for the DexYCB and HO3D datasets, respectively.

## Hand-Object Image Synthesis

We use the conditional diffsuion model to generate RGB images. We provide our model weights,

| Dataset   | Model weight                                                                                  |
| --------- | --------------------------------------------------------------------------------------------- |
| DexYCB s0 | [checkpoint](https://drive.google.com/file/d/1W4fqJciAoBNLSr_9tsMQaWAwYV2uDfn3/view?usp=sharing) |
| DexYCB s1 | \                                                                                             |
| HO3D      | [checkpoint](https://drive.google.com/file/d/14ygJYviUR5nI8XFhjsN0s0z62wk440Xu/view?usp=sharing) |

Here are the training and inference scripts,

```
# Change the dataset_name to one of [dexycb_s0, dexycb_s1, ho3d]
# Train
accelerate launch HandObjectImageSynthesizer/train.py --model_dir experiment/{dataset_name}
# Inference
accelerate launch HandObjectImageSynthesizer/inference_{dataset_name}.py --model_dir experiment/{dataset_name}--resume experiment/{dataset_name}/model/checkpoint.pt
```

## Boosted Hand-Mesh Reconstuction Baselines

Given the generated data, we mixed it with the original data to train different baselines. Cause the generated data is too large, we provide the boosted model weights,

| Baseline   | DexYCB s0                                                                                     | DexYCB s1                                                                                     | HO3D                                                                                          |
| ---------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| HandOccNet | [checkpoint](https://drive.google.com/file/d/1I-XrnoL9hnyznC0ylz2OTjyH7m-22jwQ/view?usp=sharing) | [checkpoint](https://drive.google.com/file/d/12IykKSYjgIbcdOgY7_Xv_cosp5LTpZEg/view?usp=sharing) | [checkpoint](https://drive.google.com/file/d/1xO5zEjYbHjXDPv0JHFP9p4W6XaFpmDNZ/view?usp=sharing) |
| MobRecon   | [checkpoint](https://drive.google.com/file/d/1Letj3yg7TFHjRIqC7JJeWI9LeTAXTNW_/view?usp=sharing) | [checkpoint](https://drive.google.com/file/d/1sv366DnFt4DQ_aq9O4Loxx88HfC_zUi2/view?usp=sharing) | [checkpoint](https://drive.google.com/file/d/1NXGTMhURC0_xVnstUBXb43QUe0lChnpD/view?usp=sharing) |
| H2ONet     | [checkpoint](https://drive.google.com/file/d/1Oo0ka6_GF3VaIGv-kn4uUs3IQFiKltvM/view?usp=sharing) | [checkpoint](https://drive.google.com/file/d/16secxZ2NMmZhc5o3uqggTMcdDBt_iJc5/view?usp=sharing) | [checkpoint](https://drive.google.com/file/d/1Hjeo2aPoyOFoEjxWrAsP_R7fDy0ncNT2/view?usp=sharing) |

Here is the evaluation script,

```
# Change the dataset_name to one of [dexycb_s0, dexycb_s1, ho3d] and the baseline_name to one of [handoccnet, mobrecon, h2onet]
accelerate launch HandReconstruction/test.py --model_dir experiment/{dataset_name}/{baseline_name} --resume experiment/{dataset_name}/{baseline_name}/test_model_best.pth
```

## Citation

```
@InProceedings{Xu_2024_CVPR,
    author    = {Xu, Hao and Li, Haipeng and Wang, Yinqiao and Liu, Shuaicheng and Fu, Chi-Wing},
    title     = {HandBooster: Boosting 3D Hand-Mesh Reconstruction by Conditional Synthesis and Sampling of Hand-Object Interactions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {10159-10169}
}
```

## Acknowledgments

In this project we use (parts of) the official implementations of the following works:

* [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) (Grasping pose simulation)
* [DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch) (Image generation)
* [MobRecon](https://github.com/SeanChenxy/HandMesh) (Baseline)
* [HandOccNet](https://github.com/namepllet/HandOccNet) (Baseline)
* [H2ONet](https://github.com/hxwork/H2ONet_Pytorch) (Baseline)

We thank the respective authors for open sourcing their methods.
