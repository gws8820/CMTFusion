import argparse
from torchvision import transforms
import torch
import torch.nn as nn
import time
import sys
import datetime
import torch.optim as optim
from model import CMTFusion
import utils
import losses
import torch.nn.functional as F
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dataset', type=str, default='./dataset_train/visible', help='Path of RGB dataset')
    parser.add_argument('--ir_dataset', type=str, default='./dataset_train/iwir', help='Path of IR dataset')
    parser.add_argument('--out_images', type=str, default='./dataset_train/result', help='Path of image visualization')
    parser.add_argument('--sample_interval', type=int, default=1000, help='Interval of saving image')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--b2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    args = parser.parse_args()

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # DataLoaders
    trans = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = utils.Customdataset(transform=trans, rgb_dataset=args.rgb_dataset, ir_dataset=args.ir_dataset)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print('===> Datasets loaded.')

    # Model
    fusion_model = CMTFusion().to(device)

    # Optimizer
    optimizer = optim.Adam(fusion_model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Losses
    MSE_loss = nn.MSELoss().to(device)
    loss_p = losses.perceptual_loss(device=device).to(device)
    loss_spa = losses.L_spa(device=device).to(device)
    loss_fre = losses.frequency(device=device).to(device)

    if not os.path.exists(args.out_images):
        os.makedirs(args.out_images)

    # Training Loop
    for epoch in range(args.epochs):
        fusion_model.train()
        for i, (rgb_target, ir_target) in enumerate(train_dataloader):
            optimizer.zero_grad()

            real_rgb_imgs = rgb_target.to(device)
            real_ir_imgs = ir_target.to(device)

            fake_imgs1, fake_imgs2, fake_imgs3 = fusion_model(real_rgb_imgs, real_ir_imgs)

            mse_loss = MSE_loss(fake_imgs1, real_rgb_imgs) + MSE_loss(fake_imgs1, real_ir_imgs)
            fre_loss = loss_fre(fake_imgs1, real_rgb_imgs, real_ir_imgs)
            spa_loss = 0.5 * loss_spa(fake_imgs1, real_rgb_imgs) + 0.5 * loss_spa(fake_imgs1, real_ir_imgs)
            loss_per = 0.5 * loss_p(fake_imgs1, real_rgb_imgs) + 0.5 * loss_p(fake_imgs1, real_ir_imgs)

            # Combine loss terms
            fuse_loss = mse_loss + 0.8 * spa_loss + 0.05 * loss_per + 0.02 * fre_loss

            # Convert to scalar
            fuse_loss = fuse_loss.mean()  # Use .mean() or .sum() to reduce to scalar

            # Backward pass
            fuse_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{args.epochs}, Batch {i+1}/{len(train_dataloader)}, Loss: {fuse_loss.item():.6f}")

        # Save model
        torch.save(fusion_model.state_dict(), f"./models/model_fusion_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    main()