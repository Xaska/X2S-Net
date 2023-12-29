import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model.ConvNeXt import FeatureNet as create_model
from utils import read_train_data, create_lr_scheduler, get_params_groups, train_one_epoch
from utils1 import MyDataSet, MyDataSet1, data_test, read_val_data, evaluate_test

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    img_size = 384
    val_images1_path, val_images2_path, val_images1_label, val_images2_label, val_3d_label \
        = read_val_data("../3D_dataset/test")
    data_transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.ToTensor(),
         ])
    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_dataset = MyDataSet1(images1_path=val_images1_path,
                             images2_path=val_images2_path,
                             images1_label=val_images1_label,
                             images2_label=val_images2_label,
                             edge_label1=[],
                             edge_label2=[],
                             label_3d=val_3d_label,
                             transform=data_transform,
                             img_size=img_size,
                             flag='val')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model().to(device)
    # load model weights
    model_weight_path = "./model_weight/best_model.pth"
    checkpoint = torch.load(model_weight_path)
    model.load_state_dict(checkpoint, strict=False)
    test_loss, SAC = evaluate_test(model=model,
                                 data_loader=val_loader,
                                 device=device,
                                 epoch=0,
                                 file="test"
                              )
    print("[epoch {}] test_loss: {}".format(0, round(test_loss, 4)))
    print("[epoch {}] SAC: {}".format(0, round(SAC, 4)))

if __name__ == '__main__':
    main()