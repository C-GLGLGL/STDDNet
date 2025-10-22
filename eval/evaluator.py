# Partions of this file adapted from:
# GewelsJI/VPS/eval/vps_evaluator.py
# source: https://github.com/GewelsJI/VPS
# -*- coding: utf-8 -*-
# @Time     : 2022/03/14
# @Author   : Johnson-Chou
# @Email    : johnson111788@gmail.com
# @FileName : vps_evaluator.py

import glob
import os
import cv2
import argparse
from tqdm import tqdm
import prettytable as pt
import numpy as np
from eval.metrics import Fmeasure, MAE, Smeasure, Emeasure, WeightedFmeasure, Medical
from PIL import Image

class Evaluator():
    def __init__(self, metrics, length):
        self.metrics = metrics
        self.module_map_name = {"Smeasure": "Smeasure", "wFmeasure": "WeightedFmeasure", "MAE": "MAE",
                    "adpEm": "Emeasure", "meanEm": "Emeasure", "maxEm": "Emeasure",
                    "adpFm": "Fmeasure", "meanFm": "Fmeasure", "maxFm": "Fmeasure",
                    "meanSen": "Medical", "maxSen": "Medical", "meanSpe": "Medical", "maxSpe": "Medical",
                    "meanDice": "Medical", "maxDice": "Medical", "meanIoU": "Medical", "maxIoU": "Medical"}
        self.metric_module = {}
        metric_module_list = [self.module_map_name[metric] for metric in metrics]
        metric_module_list = list(set(metric_module_list))
        
        # define measures
        for metric_module_name in metric_module_list:
            #self.metric_module[metric_module_name] = getattr(__import__("metrics", fromlist=[metric_module_name]),
            #                                            metric_module_name)(length=len(gt_pth_lst))
            if "Fmeasure" == metric_module_name:
                self.metric_module[metric_module_name] = Fmeasure(length=length)
            elif "MAE" == metric_module_name:
                self.metric_module[metric_module_name] = MAE(length=length)
            elif "Smeasure" == metric_module_name:
                self.metric_module[metric_module_name] = Smeasure(length=length)
            elif "Emeasure" == metric_module_name:
                self.metric_module[metric_module_name] = Emeasure(length=length)
            elif "WeightedFmeasure" == metric_module_name:
                self.metric_module[metric_module_name] = WeightedFmeasure(length=length)
            elif "Medical" == metric_module_name:
                self.metric_module[metric_module_name] = Medical(length=length)

    def eval(self, preds, gts, img_only=False):
        for idx in tqdm(range(preds.shape[0])):
            pred_ary = preds[idx]
            gt_ary = gts[idx]
            pred_ary = (pred_ary * 255).astype(np.uint8)
            if not img_only:
                gt_ary = (gt_ary * 255).astype(np.uint8)

            # ensure the shape of prediction is matched to gt
            if not gt_ary.shape == pred_ary.shape:
                pred_ary = cv2.resize(pred_ary, (gt_ary.shape[1], gt_ary.shape[0]))

            for module in self.metric_module.values():
                module.step(pred=pred_ary, gt=gt_ary, idx=idx)

    def get_result(self):
        res = {}
        for metric in self.metrics:
            module = self.metric_module[self.module_map_name[metric]]
            res[metric] = module.get_results()[metric]
        return res

def evaluator(gt_pth_lst, pred_pth_lst, metrics):
    module_map_name = {"Smeasure": "Smeasure", "wFmeasure": "WeightedFmeasure", "MAE": "MAE",
                       "adpEm": "Emeasure", "meanEm": "Emeasure", "maxEm": "Emeasure",
                       "adpFm": "Fmeasure", "meanFm": "Fmeasure", "maxFm": "Fmeasure",
                       "meanSen": "Medical", "maxSen": "Medical", "meanSpe": "Medical", "maxSpe": "Medical",
                       "meanDice": "Medical", "maxDice": "Medical", "meanIoU": "Medical", "maxIoU": "Medical"}
    res, metric_module = {}, {}
    metric_module_list = [module_map_name[metric] for metric in metrics]
    metric_module_list = list(set(metric_module_list))

    # define measures
    for metric_module_name in metric_module_list:
        metric_module[metric_module_name] = getattr(__import__("metrics", fromlist=[metric_module_name]),
                                                    metric_module_name)(length=len(gt_pth_lst))

    assert len(gt_pth_lst) == len(pred_pth_lst)

    # evaluator
    for idx in tqdm(range(len(gt_pth_lst))):
        gt_pth = gt_pth_lst[idx]
        pred_pth = pred_pth_lst[idx]
        # print(gt_pth, pred_pth)
        assert os.path.isfile(gt_pth) and os.path.isfile(pred_pth)

        pred_ary = cv2.imread(pred_pth, cv2.IMREAD_GRAYSCALE)
        gt_ary = cv2.imread(gt_pth, cv2.IMREAD_GRAYSCALE)

        # ensure the shape of prediction is matched to gt
        if not gt_ary.shape == pred_ary.shape:
            pred_ary = cv2.resize(pred_ary, (gt_ary.shape[1], gt_ary.shape[0]))

        for module in metric_module.values():
            module.step(pred=pred_ary, gt=gt_ary, idx=idx)

    for metric in metrics:
        module = metric_module[module_map_name[metric]]
        res[metric] = module.get_results()[metric]

    return res