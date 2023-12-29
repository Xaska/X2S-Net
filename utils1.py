import osjson
import sys
import json
import pickle
import random
import math
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
from torchvision import transforms
from Metrics import losses
import scipy.io as sio

def read_train_data(root: str):

    train_images1_path = []
    train_images2_path = []
    train_images1_label = []
    train_images2_label = []
    train_imaedge1_label = []
    train_imaedge2_label = []
    train_3d_label = []
    # print(os.getcwd())
    for file in os.listdir(os.path.join(root, "x1")):
        train_images1_path.append(os.path.join(root, "x1", file))
    for file in os.listdir(os.path.join(root, "x2")):
        train_images2_path.append(os.path.join(root, "x2", file))
    for file in os.listdir(os.path.join(root, "seg1")):
        train_images1_label.append(os.path.join(root, "seg1", file))
    for file in os.listdir(os.path.join(root, "seg2")):
        train_images2_label.append(os.path.join(root, "seg2", file))
    for file in os.listdir(os.path.join(root, "seg1_Edge")):
        train_imaedge1_label.append(os.path.join(root, "seg1_Edge", file))
    for file in os.listdir(os.path.join(root, "seg2_Edge")):
        train_imaedge2_label.append(os.path.join(root, "seg2_Edge", file))
    for file in os.listdir(os.path.join(root, "mat")):
        train_3d_label.append(os.path.join(root, "mat", file))



    return train_images1_path, train_images2_path, train_images1_label, train_images2_label, train_imaedge1_label,train_images2_label, train_3d_label


def read_val_data(root: str):
    val_images1_path = []
    val_images2_path = []
    val_images1_label = []
    val_images2_label = []
    val_3d_label = []
    for file in os.listdir(os.path.join(root, "x1")):
        val_images1_path.append(os.path.join(root, "x1", file))
    for file in os.listdir(os.path.join(root, "x2")):
        val_images2_path.append(os.path.join(root, "x2", file))
    for file in os.listdir(os.path.join(root, "seg1")):
        val_images1_label.append(os.path.join(root, "seg1", file))
    for file in os.listdir(os.path.join(root, "seg2")):
        val_images2_label.append(os.path.join(root, "seg2", file))
    for file in os.listdir(os.path.join(root, "mat")):
        val_3d_label.append(os.path.join(root, "mat", file))



    return val_images1_path, val_images2_path, val_images1_label, val_images2_label, val_3d_label
class MyDataSet(Dataset):

    def __init__(self, images1_path: list, images2_path: list, images1_label: list,
                 images2_label: list, label_3d: list, img_size: int, transform=None, ):
        self.images1_path = images1_path
        self.images2_path = images2_path
        self.images1_label = images1_label
        self.images2_label = images2_label
        self.label_3d = label_3d
        self.transform = transform
        self.trans_label = transforms.Compose([transforms.Resize((128, 128)),
                                   transforms.ToTensor()])

    def __len__(self):
        return len(self.images1_path)

    def __getitem__(self, item):


        img1 = self.binary_loader(self.images1_path[item])
        img2 = self.binary_loader(self.images2_path[item])
        label1 = self.binary_loader(self.images1_label[item])
        label2 = self.binary_loader(self.images2_label[item])

        recon_3d = sio.loadmat(self.label_3d[item])
        array_3d = recon_3d['instance']
        tensor_3d = torch.from_numpy(array_3d)
        tensor_3d = tensor_3d.unsqueeze(dim=0)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        label1 = self.trans_label(label1)
        label2 = self.trans_label(label2)


        return img1, img2, label1, label2, tensor_3d

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        return images, labels

class MyDataSet1(Dataset):

    def __init__(self, images1_path: list, images2_path: list, images1_label: list,
                 images2_label: list, edge_label1: list, edge_label2: list, label_3d: list, img_size: int, transform=None,
                 flag: str='train'):
        self.images1_path = images1_path
        self.images2_path = images2_path
        self.images1_label = images1_label
        self.images2_label = images2_label
        self.flag = flag
        if self.flag == 'train':
            self.edge_label1 = edge_label1
            self.edge_label2 = edge_label2
        self.label_3d = label_3d
        self.transform = transform
        self.trans_label = transforms.Compose([transforms.Resize((128, 128)),
                                   transforms.ToTensor()])

    def __len__(self):
        return len(self.images1_path)

    def __getitem__(self, item):


        img1 = Image.open(self.images1_path[item])
        img2 = Image.open(self.images2_path[item])

        label1 = self.binary_loader(self.images1_label[item])
        label2 = self.binary_loader(self.images2_label[item])

        recon_3d = sio.loadmat(self.label_3d[item])
        array_3d = recon_3d['instance']
        tensor_3d = torch.from_numpy(array_3d)
        tensor_3d = tensor_3d.unsqueeze(dim=0)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        label1 = self.trans_label(label1)
        label2 = self.trans_label(label2)
        if self.flag == 'train':

            edge1 = self.binary_loader(self.edge_label1[item])
            edge2 = self.binary_loader(self.edge_label2[item])
            edge1 = self.trans_label(edge1)
            edge2 = self.trans_label(edge2)
            return img1, img2, label1, label2, tensor_3d, edge1, edge2
        else:
            return img1, img2, label1, label2, tensor_3d

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    @staticmethod
    def collate_fn(batch):

        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        return images, labels
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()

    SAC = torch.zeros(1).to(device)
    HD = torch.zeros(1).to(device)
    ASD = torch.zeros(1).to(device)
    SO = torch.zeros(1).to(device)
    Disv = torch.zeros(1).to(device)
    VD = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, dynamic_ncols='auto', file=sys.stdout)
    for step, data in enumerate(data_loader):
        images1, images2, labels1, labels2, recon_3d_label, edge1, edge2 = data
        sample_num += images1.shape[0]

        pre_label1, pre_label2, recon_3d, Edge1, Edge2 = model(images1.to(device), images2.to(device))

        loss1 = Dice_loss(pre_label1, labels1.to(device)) + BCE_loss(torch.sigmoid(pre_label1), labels1.to(device))
        loss2 = Dice_loss(pre_label2, labels2.to(device)) + BCE_loss(torch.sigmoid(pre_label2), labels2.to(device))
        loss3 = losses.mae_3d_loss(recon_3d, recon_3d_label.to(device))

        loss4 = BCE_loss(torch.sigmoid(Edge1), edge1.to(device))
        loss5 = BCE_loss(torch.sigmoid(Edge2), edge2.to(device))
        loss = loss1 + loss2 + loss3 + loss4 + loss5

        loss.backward()
        mask = recon_3d > 0.5
        recon_3d[mask] = 1
        recon_3d[~mask] = 0

        for i in range(images1.shape[0]):
            reconstruction = recon_3d[i, :, :, :].squeeze()
            target = recon_3d_label[i, :, :, :].squeeze().to(device)
            SAC += losses.accuracy_3d(reconstruction.detach().int(), target)

        accu_loss += loss.detach()

        data_loader.desc = "[{}] {:.3f}, " \
                           "{:4f}, " \
                           "{:.2f},{:.6f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            SAC.item() / sample_num,
            sample_num,
            optimizer.param_groups[0]['lr']
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        optimizer.param_groups[0]['lr'] = 0.00001
    return accu_loss.item() / (step + 1), SAC.item() / sample_num

def train_one_epoch_x1_1(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()

    SAC = torch.zeros(1).to(device)
    HD = torch.zeros(1).to(device)
    ASD = torch.zeros(1).to(device)
    SO = torch.zeros(1).to(device)
    Disv = torch.zeros(1).to(device)
    VD = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, dynamic_ncols='auto', file=sys.stdout)
    for step, data in enumerate(data_loader):
        images1, images2, labels1, labels2, recon_3d_label, edge1, edge2 = data
        sample_num += images1.shape[0]

        pre_label1, recon_3d, Edge1 = model(images1.to(device))

        loss1 = Dice_loss(pre_label1, labels1.to(device)) + BCE_loss(torch.sigmoid(pre_label1), labels1.to(device))
        loss3 = losses.mae_3d_loss(recon_3d, recon_3d_label.to(device))

        loss4 = BCE_loss(torch.sigmoid(Edge1), edge1.to(device))
        loss = loss1  + loss3 + loss4

        loss.backward()
        mask = recon_3d > 0.5
        recon_3d[mask] = 1
        recon_3d[~mask] = 0

        for i in range(images1.shape[0]):
            reconstruction = recon_3d[i, :, :, :].squeeze()
            target = recon_3d_label[i, :, :, :].squeeze().to(device)
            SAC += losses.accuracy_3d(reconstruction.detach().int(), target)
        accu_loss += loss.detach()

        data_loader.desc = "[{}] {:.3f}, " \
                           "{:4f}, " \
                           "{:.2f},{:.6f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            SAC.item() / sample_num,
            sample_num,
            optimizer.param_groups[0]['lr']
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()
        # optimizer.param_groups[0]['lr'] = 0.00001
    return accu_loss.item() / (step + 1), SAC.item() / sample_num
def train_one_epoch_Edge(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    # 损失函数定义
    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    # MAE_3d = losses.mae_3d_loss()
    SAC = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, dynamic_ncols='auto', file=sys.stdout)
    for step, data in enumerate(data_loader):
        images1, images2, labels1, labels2, recon_3d_label, edge1, edge2 = data
        sample_num += images1.shape[0]

        pre_label1, pre_label2, recon_3d = model(images1.to(device), images2.to(device))

        loss1 = Dice_loss(pre_label1, labels1.to(device)) + BCE_loss(torch.sigmoid(pre_label1), labels1.to(device))
        loss2 = Dice_loss(pre_label2, labels2.to(device)) + BCE_loss(torch.sigmoid(pre_label2), labels2.to(device))
        loss3 = losses.mae_3d_loss(recon_3d, recon_3d_label.to(device))
        loss = loss1 + loss2 + loss3

        loss.backward()
        mask = recon_3d > 0.5
        recon_3d[mask] = 1
        recon_3d[~mask] = 0
        # 计算3d张量的平均表面距离（Average Surface Distance，ASD）
        for i in range(images1.shape[0]):
            reconstruction = recon_3d[i, :, :, :].squeeze()
            target = recon_3d_label[i, :, :, :].squeeze().to(device)
            SAC += losses.accuracy_3d(reconstruction.detach().int(), target)
        accu_loss += loss.detach()
        # print("Train epoch,loss,SAC,Sample_num\n")
        data_loader.desc = "[{}] {:.3f}, " \
                           "{:4f}, " \
                           "{:.2f},{:.6f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            SAC.item() / sample_num,
            sample_num,
            optimizer.param_groups[0]['lr']
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()
        # optimizer.param_groups[0]['lr'] = 0.00001
    return accu_loss.item() / (step + 1), SAC.item() / sample_num

@torch.no_grad()
def evaluate_x1_1(model, data_loader, device, epoch, file: str):
    model.eval()

    SAC = torch.zeros(1).to(device)
    HD = torch.zeros(1).to(device)
    ASD_1 = torch.zeros(1).to(device) # average_distance_gt_to_pred
    ASD_2 = torch.zeros(1).to(device) # average_distance_pred_to_gt
    SO_1 = torch.zeros(1).to(device)  # rel_overlap_gt
    SO_2 = torch.zeros(1).to(device)  # rel_overlap_pred
    Disv = torch.zeros(1).to(device)
    VD = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    if not os.path.isdir("./" + file + "/"):
        os.mkdir("./" + file + "/")
    sample_num = 0
    val_num = 0
    data_loader = tqdm(data_loader, dynamic_ncols=True, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images1, images2, labels1, labels2, recon_3d_label = data
        sample_num += images1.shape[0]

        pre_label1, recon_3d, Edge1 = model(images1.to(device))

        loss1 = Dice_loss(pre_label1, labels1.to(device)) + BCE_loss(torch.sigmoid(pre_label1), labels1.to(device))
        loss3 = losses.mae_3d_loss(recon_3d, recon_3d_label.to(device))
        loss = loss1 + loss3

        accu_loss += loss
        ## 采用SAC作为选择模型指标

        mask = recon_3d > 0.5
        recon_3d[mask] = 1
        recon_3d[~mask] = 0
        # 计算3d张量的平均表面距离（Average Surface Distance，ASD）
        for i in range(images1.shape[0]):
            reconstruction = recon_3d[i, :, :, :].squeeze()
            target = recon_3d_label[i, :, :, :].squeeze().to(device)
            # 求最佳性能指标
            SAC += losses.accuracy_3d(reconstruction.detach().int(), target)
            HD += losses.hausdorff_distance(reconstruction.detach().int().cpu().bool().numpy(),
                                            target.cpu().bool().numpy())
            SO_1 += losses.surface_overlap(reconstruction.detach().int().cpu().bool().numpy(),
                                            target.cpu().bool().numpy())[0]
            SO_2 += losses.surface_overlap(reconstruction.detach().int().cpu().bool().numpy(),
                                            target.cpu().bool().numpy())[1]
            VD += losses.volume_difference(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())
            ASD_1 += losses.calculate_asd(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())[0]
            ASD_2 += losses.calculate_asd(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())[1]
            Disv += losses.distribution_error(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())
            losses.tensor_to_obj(reconstruction.detach().int().cpu(), filename="./" + file + "/" + str(val_num) + '.obj')
            val_num += 1
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, " \
                           "SAC: {:4f}, " \
                           "HD: {:4f}, " \
                           "SO_1: {:4f}, " \
                           "SO_1: {:4f}, " \
                           "VD: {:4f}, " \
                           "ASD_1: {:4f}, " \
                           "ASD_2: {:4f}, " \
                           "Disv: {:.4f}, sample_num: {:.2f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            SAC.item() / sample_num,
            HD.item() / sample_num,
            SO_1.item() / sample_num,
            SO_2.item() / sample_num,
            VD.item() / sample_num,
            ASD_1.item() / sample_num,
            ASD_2.item() / sample_num,
            Disv.item() / sample_num,
            sample_num
        )
    return accu_loss.item() / (step + 1), SAC.item() / sample_num


@torch.no_grad()
def evaluate_Edge(model, data_loader, device, epoch, file: str):
    model.eval()

    SAC = torch.zeros(1).to(device)
    HD = torch.zeros(1).to(device)
    ASD_1 = torch.zeros(1).to(device) # average_distance_gt_to_pred
    ASD_2 = torch.zeros(1).to(device) # average_distance_pred_to_gt
    SO_1 = torch.zeros(1).to(device)  # rel_overlap_gt
    SO_2 = torch.zeros(1).to(device)  # rel_overlap_pred
    Disv = torch.zeros(1).to(device)
    VD = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    sample_num = 0
    if not os.path.isdir("./" + file + "/"):
        os.mkdir("./" + file + "/")
    val_num = 0
    data_loader = tqdm(data_loader, dynamic_ncols=True, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images1, images2, labels1, labels2, recon_3d_label = data
        sample_num += images1.shape[0]

        pre_label1, pre_label2, recon_3d = model(images1.to(device), images2.to(device))
        loss1 = Dice_loss(pre_label1, labels1.to(device)) + BCE_loss(torch.sigmoid(pre_label1), labels1.to(device))
        loss2 = Dice_loss(pre_label2, labels2.to(device)) + BCE_loss(torch.sigmoid(pre_label2), labels2.to(device))
        loss3 = losses.mae_3d_loss(recon_3d, recon_3d_label.to(device))
        loss = loss1 + loss2 + loss3

        accu_loss += loss
        ## 采用SAC作为选择模型指标

        mask = recon_3d > 0.5
        recon_3d[mask] = 1
        recon_3d[~mask] = 0
        # 计算3d张量的平均表面距离（Average Surface Distance，ASD）
        for i in range(images1.shape[0]):
            reconstruction = recon_3d[i, :, :, :].squeeze()
            target = recon_3d_label[i, :, :, :].squeeze().to(device)
            # 求最佳性能指标
            SAC += losses.accuracy_3d(reconstruction.detach().int(), target)
            HD += losses.hausdorff_distance(reconstruction.detach().int().cpu().bool().numpy(),
                                            target.cpu().bool().numpy())
            SO_1 += losses.surface_overlap(reconstruction.detach().int().cpu().bool().numpy(),
                                            target.cpu().bool().numpy())[0]
            SO_2 += losses.surface_overlap(reconstruction.detach().int().cpu().bool().numpy(),
                                            target.cpu().bool().numpy())[1]
            VD += losses.volume_difference(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())
            ASD_1 += losses.calculate_asd(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())[0]
            ASD_2 += losses.calculate_asd(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())[1]
            Disv += losses.distribution_error(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())
            losses.tensor_to_obj(reconstruction.detach().int().cpu(), filename="./" + file + "/" + str(val_num) + '.obj')
            val_num += 1
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, " \
                           "SAC: {:4f}, " \
                           "HD: {:4f}, " \
                           "SO_1: {:4f}, " \
                           "SO_1: {:4f}, " \
                           "VD: {:4f}, " \
                           "ASD_1: {:4f}, " \
                           "ASD_2: {:4f}, " \
                           "Disv: {:.4f}, sample_num: {:.2f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            SAC.item() / sample_num,
            HD.item() / sample_num,
            SO_1.item() / sample_num,
            SO_2.item() / sample_num,
            VD.item() / sample_num,
            ASD_1.item() / sample_num,
            ASD_2.item() / sample_num,
            Disv.item() / sample_num,
            sample_num
        )
    return accu_loss.item() / (step + 1), SAC.item() / sample_num
@torch.no_grad()
def evaluate_normal(model, data_loader, device, epoch, file: str):
    model.eval()

    SAC = torch.zeros(1).to(device)
    HD = torch.zeros(1).to(device)
    ASD_1 = torch.zeros(1).to(device) # average_distance_gt_to_pred
    ASD_2 = torch.zeros(1).to(device) # average_distance_pred_to_gt
    SO_1 = torch.zeros(1).to(device)  # rel_overlap_gt
    SO_2 = torch.zeros(1).to(device)  # rel_overlap_pred
    Disv = torch.zeros(1).to(device)
    VD = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    sample_num = 0
    if not os.path.isdir("./" + file + "/"):
        os.mkdir("./" + file + "/")
    val_num = 0
    data_loader = tqdm(data_loader, dynamic_ncols=True, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images1, images2, labels1, labels2, recon_3d_label = data
        sample_num += images1.shape[0]

        pre_label1, pre_label2, recon_3d, Edge1, Edge2 = model(images1.to(device), images2.to(device))
        loss1 = Dice_loss(pre_label1, labels1.to(device)) + BCE_loss(torch.sigmoid(pre_label1), labels1.to(device))
        loss2 = Dice_loss(pre_label2, labels2.to(device)) + BCE_loss(torch.sigmoid(pre_label2), labels2.to(device))
        loss3 = losses.mae_3d_loss(recon_3d, recon_3d_label.to(device))
        loss = loss1 + loss2 + loss3

        accu_loss += loss
        ## 采用SAC作为选择模型指标

        mask = recon_3d > 0.5
        recon_3d[mask] = 1
        recon_3d[~mask] = 0
        # 计算3d张量的平均表面距离（Average Surface Distance，ASD）
        for i in range(images1.shape[0]):
            reconstruction = recon_3d[i, :, :, :].squeeze()
            target = recon_3d_label[i, :, :, :].squeeze().to(device)
            # 求最佳性能指标
            SAC += losses.accuracy_3d(reconstruction.detach().int(), target)
            HD += losses.hausdorff_distance(reconstruction.detach().int().cpu().bool().numpy(),
                                            target.cpu().bool().numpy())
            SO_1 += losses.surface_overlap(reconstruction.detach().int().cpu().bool().numpy(),
                                            target.cpu().bool().numpy())[0]
            SO_2 += losses.surface_overlap(reconstruction.detach().int().cpu().bool().numpy(),
                                            target.cpu().bool().numpy())[1]
            VD += losses.volume_difference(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())
            ASD_1 += losses.calculate_asd(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())[0]
            ASD_2 += losses.calculate_asd(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())[1]
            Disv += losses.distribution_error(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())
            losses.tensor_to_obj(reconstruction.detach().int().cpu(), filename="./" + file + "/" + str(val_num) + '.obj')
            val_num += 1
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, " \
                           "SAC: {:4f}, " \
                           "HD: {:4f}, " \
                           "SO_1: {:4f}, " \
                           "SO_1: {:4f}, " \
                           "VD: {:4f}, " \
                           "ASD_1: {:4f}, " \
                           "ASD_2: {:4f}, " \
                           "Disv: {:.4f}, sample_num: {:.2f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            SAC.item() / sample_num,
            HD.item() / sample_num,
            SO_1.item() / sample_num,
            SO_2.item() / sample_num,
            VD.item() / sample_num,
            ASD_1.item() / sample_num,
            ASD_2.item() / sample_num,
            Disv.item() / sample_num,
            sample_num
        )
    return accu_loss.item() / (step + 1), SAC.item() / sample_num
@torch.no_grad()
def data_test(model, data_loader, device, epoch):

    model.eval()

    SAC = torch.zeros(1).to(device)
    HD = torch.zeros(1).to(device)
    ASD_1 = torch.zeros(1).to(device) # average_distance_gt_to_pred
    ASD_2 = torch.zeros(1).to(device) # average_distance_pred_to_gt
    SO_1 = torch.zeros(1).to(device)  # rel_overlap_gt
    SO_2 = torch.zeros(1).to(device)  # rel_overlap_pred
    Disv = torch.zeros(1).to(device)
    VD = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    sample_num = 0
    val_num = 0
    data_loader = tqdm(data_loader, dynamic_ncols=True, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images1, images2, labels1, labels2, recon_3d_label = data
        sample_num += images1.shape[0]

        pre_label1, pre_label2, recon_3d = model(images1.to(device), images2.to(device))
        loss1 = Dice_loss(pre_label1, labels1.to(device)) + BCE_loss(torch.sigmoid(pre_label1), labels1.to(device))
        loss2 = Dice_loss(pre_label2, labels2.to(device)) + BCE_loss(torch.sigmoid(pre_label2), labels2.to(device))
        loss3 = losses.mae_3d_loss(recon_3d, recon_3d_label.to(device))
        loss = loss1 + loss2 + loss3

        accu_loss += loss
        ## 采用SAC作为选择模型指标

        mask = recon_3d > 0.5
        recon_3d[mask] = 1
        recon_3d[~mask] = 0
        # 计算3d张量的平均表面距离（Average Surface Distance，ASD）
        for i in range(images1.shape[0]):
            reconstruction = recon_3d[i, :, :, :].squeeze()
            target = recon_3d_label[i, :, :, :].squeeze().to(device)
            # 求最佳性能指标
            SAC += losses.accuracy_3d(reconstruction.detach().int(), target)
            HD += losses.hausdorff_distance(reconstruction.detach().int().cpu().bool().numpy(),
                                            target.cpu().bool().numpy())
            SO_1 += losses.surface_overlap(reconstruction.detach().int().cpu().bool().numpy(),
                                            target.cpu().bool().numpy())[0]
            SO_2 += losses.surface_overlap(reconstruction.detach().int().cpu().bool().numpy(),
                                            target.cpu().bool().numpy())[1]
            VD += losses.volume_difference(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())
            ASD_1 += losses.calculate_asd(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())[0]
            ASD_2 += losses.calculate_asd(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())[1]
            Disv += losses.distribution_error(reconstruction.detach().int().cpu().bool().numpy(), target.cpu().bool().numpy())
            losses.tensor_to_obj(reconstruction.detach().int().cpu(), filename='./val/' + str(val_num) + '.obj')
            val_num += 1
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, " \
                           "SAC: {:4f}, " \
                           "HD: {:4f}, " \
                           "SO_1: {:4f}, " \
                           "SO_1: {:4f}, " \
                           "VD: {:4f}, " \
                           "ASD_1: {:4f}, " \
                           "ASD_2: {:4f}, " \
                           "Disv: {:.4f}, sample_num: {:.2f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            SAC.item() / sample_num,
            HD.item() / sample_num,
            SO_1.item() / sample_num,
            SO_2.item() / sample_num,
            VD.item() / sample_num,
            ASD_1.item() / sample_num,
            ASD_2.item() / sample_num,
            Disv.item() / sample_num,
            sample_num
        )
    return accu_loss.item() / (step + 1), SAC.item() / sample_num

