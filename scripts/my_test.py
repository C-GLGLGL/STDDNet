import os
import numpy as np
import logging
from tqdm import tqdm
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, Resize
import torch.nn.functional as F

import sys
current_path = os.path.abspath(__file__)
sys_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(sys_path)

from config import config
from lib.mamba.stddnet import STDDNet
from lib.dataloader.dataloader import get_video_dataset
from eval.evaluator import Evaluator
from eval.dice_score import dice_coeff, iou_mean, auto_data_convert
from sklearn import metrics

MODEL_TYPE = "STDDNet"

def safe_save(img, save_path, gt_path, to_resize=True):
    os.makedirs(save_path.replace(save_path.split('/')[-1], ""), exist_ok=True)
    if to_resize:
        mask = Image.open(gt_path).convert('L')
        img = img.resize(mask.size)
    img.save(save_path)

class AutoTest:
    def __init__(self, test_dataset, data_root, model_path):
        assert isinstance(test_dataset, list), "error"
        self.data_root = data_root
        self.test_dataset = test_dataset
        self.dataloader = {}
        for dst in self.test_dataset:
            self.dataloader[dst] = DataLoader(Test_Dataset(data_root, dst), batch_size=1, shuffle=False, num_workers=8)
        print('Load checkpoint:', model_path)
        self.model = Network().cuda()
        new_state = {}
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        for key, value in state_dict.items():
            new_state[key.replace('module.', '')] = value

        self.tag_dir = 'res/'+model_path.split('/')[-3]+'_'+model_path.split('/')[-2]+'/'
        self.model.load_state_dict(new_state)
        self.model.eval()

    def test(self):
        with torch.no_grad():
            for dst in self.test_dataset:
                for img, path_li in tqdm(self.dataloader[dst], desc="test:%s" % dst):
                    result = self.model(img.cuda())
                    for res, path in zip(result, path_li[1:]):
                        npres = res.squeeze().cpu().numpy()
                        safe_save(Image.fromarray((npres * 255).astype(np.uint8)),
                                  path[0].replace(self.data_root, self.tag_dir).replace(".jpg", ".png").replace('Frame', ''))

class VPSTest:
    def __init__(self, data_root, test_dataset, model_path):
        self.data_root = data_root
        self.test_dataset = test_dataset
        self.dataloader = {}
        for dst in self.test_dataset:
            eval_dataset = get_video_dataset(dst, 'eval')
            self.dataloader[dst] = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

        ### Load Model ###
        if MODEL_TYPE == "STDDNet":
            model = STDDNet(f_num=config.video_time_clips, img_size=config.size, mlp_ratio=2.0)

        model_state = torch.load(model_path)
        model.load_state_dict(model_state)
        self.model = model.cuda(device=device_ids[0]).eval()
        self.tag_dir = '/'.join(model_path.split('/')[1:-1]) + '/'
    
    def test(self, eval_on=True):
        with torch.no_grad():
            for dst in self.test_dataset:
                logging.info("Test dataset " + str(dst) + ": ")
                tot_dice = 0.
                # tot_iou = 0.
                tot_mae = 0.
                evaluator = Evaluator(config.metric_list, 1*(config.video_time_clips+1))
                mean_case_score_list, max_case_score_list = [], []
                nums = 0
                for i, (images, gts, path_li) in enumerate(self.dataloader[dst], start=1):
                    images = images.cuda(device=device_ids[0])

                    if MODEL_TYPE == "STDDNet":
                        preds = self.model(images, mode='eval', sdpm_on=True, ddfe_on=True)
                        # preds = self.model(images)
                    else:
                        preds = self.model(images)
                        preds = torch.sigmoid(preds)
                    gts = gts.reshape(-1, gts.shape[3], gts.shape[4])

                    if config.tf_img_only:
                        preds = F.interpolate(preds, size=(gts.shape[1], gts.shape[2]), mode='bilinear', align_corners=False)

                    for res, path in zip(preds, path_li):
                        npres = res.squeeze().cpu().numpy()
                        save_path = '/' + path[0][0].replace(self.data_root, self.tag_dir).replace(".jpg", ".png").replace('Frame/', '')
                        gt_path = path[1][0]
                        safe_save(Image.fromarray((npres * 255).astype(np.uint8)), save_path, gt_path)

                    # Evaluation Dice/IoU/MAE #
                    if eval_on:
                        if not config.tf_img_only:
                            tot_dice += dice_coeff(preds.squeeze(), gts.to(preds.device), preds.device).item()
                            m = auto_data_convert(preds.squeeze()).astype(int)
                            t = auto_data_convert(gts).astype(int)
                            # tot_iou += iou_mean(m, t)
                            tot_mae += metrics.mean_absolute_error(t, m)
                        else:
                            tot_dice += dice_coeff(preds.squeeze(), gts.float().to(preds.device), preds.device).item()
                        nums += 1

                        preds = preds.squeeze().cpu().detach().numpy()
                        gts = gts.squeeze().cpu().numpy()
                        evaluator.eval(preds, gts, config.tf_img_only)
                
                if eval_on:
                    mean_dice = tot_dice / nums
                    # mean_iou = tot_iou / nums
                    mean_mae = tot_mae / nums
                    print("Dice coeff: ", mean_dice)
                    logging.info("Dice coeff: " + str(mean_dice))
                    print("MAE: ", mean_mae)
                    logging.info("MAE: " + str(mean_mae))


                    result = evaluator.get_result()
                    mean_score_ind, max_score_ind = [], []
                    mean_score_list, max_score_list = [], []
                    for i, (name, value) in enumerate(result.items()):
                        if 'max' in name or 'mean' in name:
                            if 'max' in name:
                                max_score_list.append(value)
                                max_score_ind.append(i)
                            else:
                                mean_score_list.append(value)
                                mean_score_ind.append(i)
                        else:
                            mean_score_list.append([value]*256)
                            mean_score_ind.append(i)

                    # calculate all the metrics at frame-level
                    max_case_score_list.append(max_score_list)
                    mean_case_score_list.append(mean_score_list)
                    max_case_score_list = np.mean(np.array(max_case_score_list), axis=0)
                    mean_case_score_list = np.mean(np.array(mean_case_score_list), axis=0)
                    case_score_list = []
                    for index in range(len(config.metric_list)):
                        real_max_index = np.where(np.array(max_score_ind) == index)
                        real_mean_index = np.where(np.array(mean_score_ind) == index)
                        if len(real_max_index[0]) > 0:
                            case_score_list.append(max_case_score_list[real_max_index[0]].max().round(5))
                        else:
                            case_score_list.append(mean_case_score_list[real_mean_index[0]].mean().round(5))
                    final_score_list = ['{:.5f}'.format(case) for case in case_score_list]
                    print([config.metric_list[i] + ": " + final_score_list[i] for i in range(len(final_score_list))])
                    logging.info([config.metric_list[i] + ": " + final_score_list[i] for i in range(len(final_score_list))])


gpu_id = config.gpu_id
if ',' in gpu_id:
    device_ids = gpu_id.split(',')
    device_ids = [int(idx) for idx in device_ids]
else:
    device_ids = [int(gpu_id)]
device = torch.device('cuda:{}'.format(device_ids[0]) if torch.cuda.is_available() else 'cpu')
print('USE GPU: ', gpu_id)

if __name__ == "__main__":

    model_path = "/media/cgl/STDDNet/experiments/2025-1021-094853/ckpt_epoch_2.pth"      # PATH of saved model

    save_dir = '/' + '/'.join(model_path.split('/')[1:-1]) + '/'
    # logging
    logging.basicConfig(filename=os.path.join(save_dir,'test.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.getLogger(__name__)

    vpstest = VPSTest(config.dataset_root, 
                    ['TestEasyDataset/Seen', 'TestEasyDataset/Unseen', 'TestHardDataset/Seen', 'TestHardDataset/Unseen'],   # for SUN-SEG
                    # ['TestDataset'],                                                                                      # for CVC-ClinicDB, self-defined
                    model_path)

    vpstest.test(eval_on=True)
