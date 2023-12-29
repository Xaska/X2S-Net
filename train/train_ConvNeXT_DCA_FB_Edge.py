import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils1 import MyDataSet, MyDataSet1
from model.ConvNeXT_DCA_FB_Edge import FeatureNet as create_model
from utils1 import read_train_data, read_val_data, train_one_epoch, evaluate_normal
from utils import create_lr_scheduler, get_params_groups
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    train_images1_path, train_images2_path, train_images1_label, train_images2_label, train_imaedge1_label,train_imaedge2_label,train_3d_label \
        = read_train_data(args.train_data_path)
    val_images1_path, val_images2_path, val_images1_label, val_images2_label, val_3d_label \
        = read_val_data(args.val_data_path)
    img_size = 384
    data_transform = {
        "train": transforms.Compose([transforms.Resize((img_size, img_size)),
                                     # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.2),
                                     # transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)], p=0.3),
                                     # transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                                     transforms.ToTensor(),
                                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
        "val": transforms.Compose([transforms.Resize((img_size, img_size)),
                                   transforms.ToTensor(),
                                   # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}

    train_dataset = MyDataSet1(images1_path=train_images1_path,
                              images2_path=train_images2_path,
                              images1_label=train_images1_label,
                              images2_label=train_images2_label,
                              edge_label1=train_imaedge1_label,
                              edge_label2=train_imaedge2_label,
                              label_3d=train_3d_label,
                              transform=data_transform["train"],
                              img_size=img_size)

    val_dataset = MyDataSet1(images1_path=val_images1_path,
                             images2_path=val_images2_path,
                             images1_label=val_images1_label,
                             images2_label=val_images2_label,
                             edge_label1=[],
                             edge_label2=[],
                             label_3d=val_3d_label,
                             transform=data_transform["val"],
                             img_size=img_size,
                             flag='val')
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)
    model = create_model().to(device)


    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)
    best_acc = 0.
    start_epoch = 0

    best_acc = 0.
    start_epoch = 0
    if args.RESUME:
        checkpoint_conv = torch.load(r"./model_weight/convnext_xlarge_22k_1k_384_ema.pth", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint_conv['model'], strict=False)
    if args.RESUME == False:
        path_checkpoint = "./model_weight_Edge/checkpoint/ckpt_best_70.pth"
        print("model continue train")
        checkpoint = torch.load(path_checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_schedule'])
        model.load_state_dict(checkpoint["net"], strict=True)
        del checkpoint


    for epoch in range(start_epoch + 1, args.epochs + 1):

        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate_normal(model=model,
                                          data_loader=val_loader,
                                          device=device,
                                          epoch=epoch,
                                          file="val_Edge")

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            if not os.path.isdir("./model_weight_Edge"):
                os.mkdir("./model_weight_Edge")
            torch.save(model.state_dict(), "./model_weight_Edge/best_model.pth")
            print("Saved epoch{} as new best model".format(epoch))
            best_acc = val_acc

        if epoch % 10 == 0:
            print('epoch:', epoch)
            print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'lr_schedule': lr_scheduler.state_dict()
            }
            if not os.path.isdir("./model_weight_Edge/checkpoint"):
                os.mkdir("./model_weight_Edge/checkpoint")
            torch.save(checkpoint, './model_weight_Edge/checkpoint/ckpt_best_%s.pth' % (str(epoch)))

        #add loss, acc and lr into tensorboard
        print("[epoch {}] accuracy: {}".format(epoch, round(val_acc, 3)))

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    import json
    path = "./data_path.json"
    with open(path, "r") as f:
        row_data = json.load(f)


    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--RESUME', type=bool, default=False)

    # parser.add_argument('--train_data_path', type=str, default="../../home/dell/wzt/dataset/train")
    # parser.add_argument('--val_data_path', type=str, default="../../home/dell/wzt/dataset/validation")

    parser.add_argument('--train_data_path', type=str, default=row_data["train_data_path"])
    parser.add_argument('--val_data_path', type=str, default=row_data["val_data_path"])

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    print("=============")
    if torch.cuda.is_available():
        print('CUDA 版本:', torch.version.cuda)
        print('设备名称:', torch.cuda.get_device_name(0))
        print('当前设备:', torch.cuda.current_device())
        print(torch.cuda.is_available())
        print(torch.__version__)


    else:
        print('CUDA 不可用')

    main(opt)