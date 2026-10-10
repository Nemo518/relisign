### Dataset structure
For simplicity, our datasets are structured in the following way:
```
/Uni-Sign/dataset/
├── CSL_News
│   ├── rgb_format # mp4 format
│   │   ├── Common-Concerns_20201113_0-287_14330.mp4 
│   │   ├── Common-Concerns_20201113_1562-2012_239580.mp4
│   │   └── ...
│   │ 
│   └── pose_format # pkl format
│       ├── Common-Concerns_20201113_0-287_14330.pkl 
│       ├── Common-Concerns_20201113_1562-2012_239580.pkl
│       └── ...
│      
├── CSL_Daily
│   ├── Archive/ 
│   ├── label/ 
│   ├── sentence-crop # mp4 format
│   │   ├── S005870_P0006_T00.mp4
│   │   ├── S005870_P0009_T00.mp4
│   │   └── ...
│   │ 
│   └── pose_format # pkl format
│       ├── S005870_P0006_T00.pkl
│       ├── S005870_P0009_T00.pkl
│       └── ...
│      
├── WLASL
│   ├── rgb_format # mp4 format
│   │   ├── train/
│   │   ├── val/
│   │   └── test
│   │       ├── 64550.mp4
│   │       ├── 64551.mp4
│   │       └── ...
│   │   
│   └── pose_format # pkl format
│       ├── train/
│       ├── val/
│       └── test
│           ├── 64550.pkl
│           ├── 64551.pkl
│           └── ...
```

#### Note: 
* You first need to download the [mt5-base](https://huggingface.co/google/mt5-base) weights, and place them in the `./pretrained_weight/mt5-base`.
* Download the [CSL-News](https://huggingface.co/datasets/ZechengLi19/CSL-News/tree/main), [CSL-Daily](https://ustc-slr.github.io/datasets/2021_csl_daily/), [WLASL](https://github.com/dxli94/WLASL), [How2Sign](https://how2sign.github.io/) and [OpenASL](https://github.com/chevalierNoir/OpenASL) datasets based on your requirements.
* If the `sentence-crop` folder is missing in the CSL-Daily dataset, please refer to Issue [#7](https://github.com/ZechengLi19/Uni-Sign/issues/7) for guidance.
* The `pose_format` data of CSL-News can be downloaded from [here](https://huggingface.co/datasets/ZechengLi19/CSL-News_pose), while the RGB video data are provided at [here](https://huggingface.co/datasets/ZechengLi19/CSL-News).
* The `pose_format` folders for the CSL-Daily, WLASL, How2Sign, and OpenASL datasets can be downloaded from [here](https://huggingface.co/ZechengLi19/Uni-Sign). These pose data are extracted using RTMPose from MMPose. The pose extraction process is provided in the [code](../demo/pose_extraction.py).
* For How2Sign and OpenASL, the pose data are split into multiple files (e.g., `how2sign_pose_format.zip.*`). You can merge and extract them using:
```bash
# For How2Sign dataset
cat how2sign_pose_format.zip.* > how2sign_pose_format.zip
unzip how2sign_pose_format.zip
# For OpenASL dataset
cat openasl_pose_format.zip.* > openasl_pose_format.zip
unzip openasl_pose_format.zip
```
* The Uni-Sign checkpoints can be found [here](https://huggingface.co/ZechengLi19/Uni-Sign).
* If the datasets or mt5 checkpoint are stored in different paths, you can modify the `config.py` file to specify the new paths.
