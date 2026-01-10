'''
Descripttion: 
version: 
Author: Luckie
Date: 2022-08-05 16:05:46
LastEditors: Luckie
LastEditTime: 2022-08-05 16:07:19
'''
import SimpleITK as sitk
import os
import numpy as np
from medpy import metric
from glob import glob

from tqdm import tqdm

def calculate_metric(gt, pred, cal_hd95=False):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        if cal_hd95:
            hd95 = metric.binary.hd95(pred, gt)
        else:
            hd95 = 0.0
        return np.array([dice, hd95])
    else:
        return np.zeros(2)





def cal_metrics_two_folders(gt_path,pred_path):
    pred_list = glob(pred_path + "*nii.gz")
    save_result_file = pred_path+"/res.txt"
    print("total pred file:",len(pred_list))
    total_metrics = np.zeros((2,len(pred_list),2))
    result_dict = {}
    sorted(pred_list)
    with open(save_result_file, 'w') as f:
        for i,pred_file in enumerate(tqdm(pred_list)):
            _, pred_name = os.path.split(pred_file)
            gt_file = gt_path + pred_name
            assert os.path.isfile(gt_file),"gt file does not exists"
            pred =sitk.GetArrayFromImage(sitk.ReadImage(pred_file))
            gt =sitk.GetArrayFromImage(sitk.ReadImage(gt_file))
            print("pred file:",pred_file)
            print(f"pred unique:{np.unique(pred)},gt unique:{np.unique(gt)}")
            # gt[(gt!=2) & (gt!=3)] = 0
            # gt[gt==2] = 1
            # gt[gt==3] = 2
            # gt[gt!=5] = 0
            # gt[gt==5] = 1
            #metric = calculate_metric(gt,pred,cal_hd95=True)
            for j in range(1,3):
                metric = calculate_metric(gt==j,pred==j,cal_hd95=True)
                print(metric)
                total_metrics[j-1,i] = metric
                f.writelines(f"name:{pred_name}, dice:{metric[0]}, hd:{metric[1]} \n")
        #total_metrics /= len(pred_list)
        f.writelines(f"mean dice:{total_metrics[0,:,0].mean()},"\
                     f"mean hd:{total_metrics[0,:,1].mean()} \n")
        try:
            f.writelines(f"mean dice:{total_metrics[1,:,0].mean()},"\
                         f"mean hd:{total_metrics[1,:,1].mean()} \n")
        except:
            print("only have one class")

        print(f"dice:{total_metrics[0,:,0].mean()},"\
            f"hd95:{total_metrics[0,:,1].mean()}")
        try:
            print(f"dice:{total_metrics[1,:,0].std()},"\
                f"hd95:{total_metrics[1,:,1].std()}")
        except:
            print("only have one class")
        # print(f"dice:{total_metrics[1,:,0].mean()},"\
        #       f"hd95:{total_metrics[1,:,1].mean()}")
        # print(f"dice:{total_metrics[1,:,0].std()},"\
        #       f"hd95:{total_metrics[1,:,1].std()}")


if __name__ == "__main__":
    gt_path = ""
    pred_path = ""
    print(pred_path)
    cal_metrics_two_folders(gt_path,pred_path)