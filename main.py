"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from sklearn.manifold import TSNE
import cv2
import PIL

import os
import argparse
import datetime
import logging

from models import *
from utils import progress_bar

train_start_time = datetime.datetime.now()
train_chk_directory = f"./checkpoint/{train_start_time.strftime('%Y%m%d_%H%M')}/"
if not os.path.exists(train_chk_directory):
    os.makedirs(train_chk_directory)

logging.basicConfig(
    filename=f"{train_chk_directory}/log.txt",
    format="%(asctime)s %(levelname)s %(message)s",  # " ".join([f"%({name})s" for name in ["asctime", "level", "message"]]),
    level=logging.DEBUG,
)
logging.getLogger().addHandler(logging.StreamHandler())

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def convertColor(img):
    # print(type(img), img.shape)
    img = img.numpy()
    img = np.moveaxis(img, 0, -1)
    # print(img.shape)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    ycrcb = np.moveaxis(ycrcb, -1, 0)
    return torch.tensor(ycrcb)


# Data
logging.info("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(convertColor),
        transforms.Normalize((0.4082, 0.4780, 0.5170), (0.2819, 0.0638, 0.0702)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(convertColor),
        transforms.Normalize((0.4082, 0.4780, 0.5170), (0.2819, 0.0638, 0.0702)),
    ]
)

num_classes = 10
dataset_to_use = (
    torchvision.datasets.CIFAR10 if num_classes == 10 else torchvision.datasets.CIFAR100
)

trainset = dataset_to_use(
    root="./data", train=True, download=True, transform=transform_train
)
logging.info(trainset)
len(f"length of trainset.classes = {len(trainset.classes)}")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
logging.info(f"trainloader.batch_size = {trainloader.batch_size}")


def calculate_whiten():
    train_imgs = [img for batch, (img, labels) in enumerate(trainloader)]
    train_imgs = np.concatenate(train_imgs)
    train_imgs = np.moveaxis(train_imgs, 1, -1)
    train_imgs = np.reshape(
        train_imgs,
        (
            train_imgs.shape[0] * train_imgs.shape[1] * train_imgs.shape[2],
            train_imgs.shape[3],
        ),
    )
    print(train_imgs.shape)
    print(
        np.mean(train_imgs[:, 0]), np.mean(train_imgs[:, 1]), np.mean(train_imgs[:, 2])
    )
    print(np.std(train_imgs[:, 0]), np.std(train_imgs[:, 1]), np.std(train_imgs[:, 2]))


testset = dataset_to_use(
    root="./data", train=False, download=True, transform=transform_test
)
logging.info(f"{testset.data.shape} test data shape")

testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

# Model
print("==> Building model..")
# model = VGG('VGG19')
# model = ResNet18()
# model = ResNet34()
model = ResNet50(
    num_classes=num_classes,
    use_oriented_maps_v1="power",
    oriented_maps_v1_kernel_size=11,
    use_oriented_maps_bottleneck="phase",
    oriented_maps_bottleneck_kernel_size=11,
    use_depthwise_maxpool=True,
)
# model = ResNet101(
#     num_classes=num_classes,
#     use_oriented_maps_v1="power",  # "power",
#     use_oriented_maps_bottleneck="phase",  # "phase",
#     use_depthwise_maxpool=True,
# )
# # model = PreActResNet18()
# model = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()

model = model.to(device)
if device == "cuda":
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True
else:
    net = model

model.train_oriented_maps(False)

logging.info(str(model))

criterion = nn.CrossEntropyLoss()
logging.info(f"{str(criterion)}")

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
logging.info(f"{str(optimizer)}")

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
logging.info(f"{type(scheduler).__name__}:{scheduler.T_max} {scheduler.eta_min}")


def plot_test_tsne(testloader: torch.utils.data.DataLoader):
    with torch.no_grad():
        class_outputs = [
            net(inputs.to(device)).cpu().numpy()
            for _, (inputs, _) in enumerate(testloader)
        ]
        class_outputs = np.concatenate(class_outputs)

    # create tsne projector
    tsne = TSNE(
        n_components=2,
        perplexity=num_classes * 2,
        early_exaggeration=1,
        metric="cosine",
        learning_rate="auto",
        init="random",
        n_iter=1000,
        verbose=3,
    )

    projected_outputs = tsne.fit_transform(class_outputs)

    return (class_outputs, projected_outputs)


# try out the plotting
# test_images = testset.data
# projected_outputs = plot_test_tsne(testloader)
# print(f"{projected_outputs.shape} {test_images.shape}")


# if args.resume:
#     # Load checkpoint.
#     print("==> Resuming from checkpoint..")
#     assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
#     checkpoint = torch.load("./checkpoint/ckpt.pth")
#     net.load_state_dict(checkpoint["net"])
#     best_acc = checkpoint["acc"]
#     start_epoch = checkpoint["epoch"]


# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        logging.info(
            "Batch %d: Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (
                batch_idx,
                train_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
            )
        )


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            logging.info(
                "Test Batch %d: Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    batch_idx,
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                )
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        logging.info("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        torch.save(state, f"{train_chk_directory}/ckpt.pth")
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 100):

    train(epoch)
    test(epoch)
    
    class_positions, projected_positions = plot_test_tsne(testloader)

    logging.info(f"projected_positions {projected_positions.shape} shape")

    np.save(f"{train_chk_directory}/{epoch:03d}-class.npy", class_positions)
    logging.info(f"saved class positions to {epoch:03d}-class.npy")

    np.save(f"{train_chk_directory}/{epoch:03d}-tsne.npy", projected_positions)
    logging.info(f"saved tsne projection to {epoch:03d}-tsne.npy")

    scheduler.step()


post_train = False
if post_train:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    model.train_oriented_maps(True)

    for epoch in range(start_epoch + 100, start_epoch + 200):
        train(epoch)
        test(epoch)
        scheduler.step()
