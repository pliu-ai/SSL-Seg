"test for conditional model"
import os 
import argparse
import torch
from glob import glob  
import SimpleITK as sitk 
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from typing import OrderedDict
import yaml
import shutil

from networks.net_factory_3d import net_factory_3d
from val_3D import test_single_case,calculate_metric
from batchgenerators.utilities.file_and_folder_operations import save_json

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='test_config_new.yaml', help='model_name')
parser.add_argument('--gpu', type=str, default='0',help='gpu id for testing')
parser.add_argument('--model_path', type=str,
                    default='../model/BCV_4_C3PS_test/unet_3D/'\
                            'model2_iter_15800_dice_0.7323.pth')

class_id_name_dict = {
    'MMWHS':['MYO', 'LA', 'LV', 'RA', 'AA', 'PA', 'RV'],
    'BCV':['Spleen', 'Right Kidney', 'Left Kidney','Liver','Pancreas'],
    'LA':['LA']
}

def predict_multiprocess():
    pass

def test_all_case_condition(net, test_list="full_test.list", num_classes=4, 
                        patch_size=(48, 160, 160), stride_xy=32, stride_z=24, 
                        condition=-1, method="regular",
                        cal_metric=True,
                        save_prediction=False,
                        prediction_save_path='./',
                        cut_upper=1000,
                        cut_lower=-1000,
                        con_list=None):
    if os.path.isdir(test_list):
        image_list = glob(test_list+"*.nii.gz")
    else:
        with open(test_list, 'r') as f:
            image_list = [img.replace('\n','') for img in f.readlines()]
    print("Total test images:",len(image_list))
    all_scores = OrderedDict() # for save as json
    all_scores['all'] = []
    all_scores['mean'] = OrderedDict()
    
    print("Validation begin")
    if con_list:
        condition_list = con_list 
    else:
        condition_list = [i for i in range(1,num_classes)]
    total_metric = np.zeros((len(condition_list), 2))
    img_num = np.zeros((num_classes-1,1))
    for i, image_path in enumerate(tqdm(image_list)):
        print(f"===========>processing {image_path}")
        res_metric = OrderedDict()
        if len(image_path.strip().split()) > 1:
            image_path, mask_path = image_path.strip().split()
        else: 
            mask_path = image_path.replace('img','label')
        if cal_metric:
            assert os.path.isfile(mask_path),"invalid mask path error"
        
        # use for save json results
        res_metric['image_path'] = image_path 
        res_metric['mask_path'] = mask_path
        
        image_sitk = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image_sitk)
        shape = image.shape
        
        if cal_metric: # whether calculate metrics
            label = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        else:
            label = np.zeros_like(image)
        np.clip(image,cut_lower,cut_upper,out=image)
        image = (image - image.mean()) / image.std()

        prob_map = np.ones((num_classes,shape[0],shape[1],shape[2]))
        for i, con_index in enumerate(condition_list):
            pred_con, prob = test_single_case(net, image, stride_xy, stride_z, 
                                               patch_size, 
                                               num_classes=num_classes, 
                                               condition=con_index, 
                                               method=method,
                                               return_scoremap=True)
            prob_map[0,:][prob[0]<prob_map[0,:]] = prob[0][prob[0]<prob_map[0,:]]
            prob_map[con_index,:] = prob[1]
            metric = calculate_metric(label == con_index, pred_con)
            print(f"con:{con_index}, metric:{metric}")
        #prob_map[0] /= len(condition_list)
        prediction = np.argmax(prob_map,0)
        if cal_metric:
            each_metric = np.zeros((num_classes-1, 2))
            for i,con_index in enumerate(condition_list):
                metrics = calculate_metric(label == con_index, prediction == con_index)
                
                print(f"class:{class_name_list[con_index-1]}, metric:{metrics}")
                res_metric[class_name_list[con_index-1]] = {
                    'Dice':metrics[0],'HD95':metrics[1]
                }
                img_num[con_index-1]+=1
                total_metric[con_index-1, :] += metrics
                each_metric[con_index-1, :] += metrics
            res_metric['Mean'] = {
                'Dice':each_metric[:,0].mean(),'HD95':each_metric[:,1].mean()
            }
            all_scores['all'].append(res_metric)
        # save prediction
        if save_prediction: 
            spacing = image_sitk.GetSpacing()
            pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
            pred_itk.SetSpacing(spacing)
            _,image_name = os.path.split(image_path)
            sitk.WriteImage(pred_itk, prediction_save_path+image_name.replace(".nii.gz","_pred.nii.gz"))

    mean_metric = total_metric / img_num
    for i in range(1, num_classes):
        all_scores['mean'][class_name_list[i-1]] = {
            'Dice':mean_metric[i-1][0],
            'HD95': mean_metric[i-1][1]
        }
    all_scores['mean']['mean']={
        'Dice':mean_metric[:,0].mean(),
        'HD95':mean_metric[:,1].mean()
    }
    
    save_json(all_scores,prediction_save_path+"/Results.json")
    print("***************************validation end**************************")
    return mean_metric

def main(args, config):
    model = net_factory_3d("unet_3D_condition", in_chns=1, class_num=2).cuda()
    model_state_dict = torch.load(config['model_checkpoint'])
    model.load_state_dict(model_state_dict)
    dataset_name = config['dataset_name']
    dataset_config = config['DATASET'][dataset_name]
    cut_upper = dataset_config['cut_upper']
    cut_lower = dataset_config['cut_lower']
    test_list = dataset_config['test_list']
    patch_size = (96,160,160)
    model.eval()


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model_path = args.model_path
    root_path,_ = os.path.split(model_path)
    config_file_list = glob(root_path+"/*yaml")
    sorted(config_file_list)
    config_file = config_file_list[-1]
    print(f"===========> using config file: {os.path.split(config_file)[1]}")
    config = yaml.safe_load(open(config_file, 'r'))
    method_name = config['method']

    dataset_name = config['dataset_name']
    class_name_list = class_id_name_dict[dataset_name]
    dataset_config = config['DATASET'][dataset_name]
    cut_upper = dataset_config['cut_upper']
    cut_lower = dataset_config['cut_lower']
    num_classes = dataset_config['num_classes']
    
    pred_save_path = f"{root_path}/Prediction_con/"
    if os.path.exists(pred_save_path):
        shutil.rmtree(pred_save_path)
    os.makedirs(pred_save_path)
    model = net_factory_3d("unet_3D_condtion_decoder", in_chns=1, class_num=2).cuda()
    model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    test_list = dataset_config['test_list']
    patch_size = config['DATASET']['patch_size']
    model = model.cuda()
    model.eval()
    metrics = test_all_case_condition(
        model,
        test_list=test_list,
        num_classes=num_classes, 
        patch_size=patch_size,
        stride_xy=64, 
        stride_z=64,
        condition=1,
        cut_lower=cut_lower,
        cut_upper=cut_upper,
        con_list = [1,2,3,4,5],
        save_prediction=True,
        prediction_save_path=pred_save_path
)
