Weakly Supervised Object Localization via Transformer with Implicit
Spatial Calibration
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

# Description

- NBDev version of Weakly Supervised Object Localization via Transformer
  with Implicit Spatial Calibration with added features  
- LR Scheduler adpated from YOLOX

## Install

    pip install scm_nbdev

## How to train

# CUB

python tools_cam/train_cam.py –config_file
./configs/CUB/deit_scm_small_patch16_224.yaml –lr 5e-5

# ILSVRC

python tools_cam/train_cam.py –config_file
./configs/ILSVRC/deit_scm_small_patch16_224.yaml –lr 1e-6

## How to inference

python tools_cam/test_cam.py –config_file {config_file} –lr {lr}
