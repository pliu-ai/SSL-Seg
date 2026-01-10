import numpy as np
from tqdm import tqdm
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import zoom
from utils import binary
from dataset.dataset_old import BaseDataSets
from trainer.semi_trainer_2D import SemiSupervisedTrainer2D


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='train_config_2d.yaml', help='training configuration')

def calculate_metric_percase(gt, pred, cal_hd95=False, 
                             cal_asd=True, spacing=None):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = binary.dc(pred, gt)
        if cal_hd95:
            hd95 = binary.hd95(pred, gt)
        else:
            hd95 = 0.0
        if cal_asd:
            asd = binary.asd(pred, gt)
        else:
            asd = 0.0
        return np.array([dice, asd])
    else:
        return np.array([0.0, 150])



def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def evaluate_model(model, dataset, num_classes, patch_size=(224,224)):
    metric_list = 0.0
    dataloader_val = DataLoader(dataset, batch_size=1, 
                                    shuffle=False,num_workers=1)
    iterations = 0
    for i_batch, sampled_batch in enumerate(tqdm(dataloader_val)):
        iterations += 1
        if iterations > 1:
            break 
        volume_batch, label_batch = ( 
                sampled_batch['image'], sampled_batch['label']
            )
        metric_i = test_single_volume(volume_batch,label_batch,model,
                                      classes=num_classes, 
                                      patch_size=patch_size)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(dataset)
    for class_i in range(num_classes-1):
        print(
            f'info/val_{class_i+1}_dice',
            metric_list[class_i, 0]
        )
        print(
            f'info/val_{class_i+1}_asd',
            metric_list[class_i, 1]
        )
    mean_dsc = np.mean(metric_list, axis=0)[0]
    mean_asd = np.mean(metric_list,axis=0)[1]
    print(f"mean dice: {mean_dsc}, mean asd: {mean_asd}")

    


if __name__ == '__main__':
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    snapshot_path = "../model/{}_{}_{}_{}_{}_{}/{}".format(
        config['dataset_name'], 
        config['DATASET']['labeled_num'], 
        config['method'], 
        config['exp'],
        config['optimizer_type'],
        config['optimizer2_type'],
        config['backbone']   
    )
    trainer = SemiSupervisedTrainer2D(config=config, 
                                      root_path=config['root_path'],
                                      output_folder=snapshot_path)
    trainer.initialize_network()
    trainer.initialize()
    trainer.load_checkpoint("/data1/liupeng/From_DGX/semi-supervised_segmentation/SSL4MIS-master/model/BCV/Cross_Teaching_Between_CNN_Transformer_4/unet/model1_iter_26400_dice_0.6822.pth")
    trainer.evaluation(model=trainer.model, model_name="vit")
    # patch_size = (448,448)
    # root_path = config['root_path']
    # num_classes = 8
    
    
    # dataset = BaseDataSets(base_dir=root_path, split="val")
    # evaluate_model(trainer.model2,dataset,num_classes,patch_size)