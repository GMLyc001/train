import os
from generator import Generator
from discriminator import Discriminator
from utils.dataset import ImagesDataset
from training.train_config import config
from torch.utils.data import DataLoader
from torchgan.losses import MinimaxGeneratorLoss, MinimaxDiscriminatorLoss
from torch.optim import Adam
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

def main():
    dataset = ImagesDataset(images_path=r"E:\TRYRYRY\.venv\datasets\picture384\*.jpg")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    generator = Generator()
    discriminator = Discriminator()

    optimizer_gen = Adam(generator.parameters(), lr=config['learning_rate_gen'], betas=(config['beta1'], config['beta2']))
    optimizer_disc = Adam(discriminator.parameters(), lr=config['learning_rate_disc'], betas=(config['beta1'], config['beta2']))

    generator_loss = MinimaxGeneratorLoss()
    discriminator_loss = MinimaxDiscriminatorLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    writer = SummaryWriter('./torchgan_logs')

    for epoch in range(config['epochs']):
        start_time = time.time()
        total_generator_loss = 0.0
        total_discriminator_loss = 0.0

        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)

            # 生成噪声并生成假图像
            noise = torch.randn(config['batch_size'], config['noise_dim'], 1, 1, device=device)
            fake_images = generator(noise)

            # 清除之前的梯度
            optimizer_gen.zero_grad()
            optimizer_disc.zero_grad()

            # 计算判别器损失
            real_output = discriminator(real_images)
            fake_output = discriminator(fake_images)
            real_loss = discriminator_loss(real_output, torch.ones_like(real_output))
            fake_loss = discriminator_loss(fake_output, torch.zeros_like(fake_output))
            total_disc_loss = real_loss + fake_loss

            # 反向传播和优化判别器
            total_disc_loss.backward()
            optimizer_disc.step()

            # 计算生成器损失
            fake_output = discriminator(fake_images)
            g_loss = generator_loss(fake_output)
            total_generator_loss += g_loss.item()

            # 反向传播和优化生成器
            g_loss.backward()
            optimizer_gen.step()

            if i == 0 and epoch % 10 == 0:
                grid = vutils.make_grid(fake_images, padding=2, normalize=True)
                writer.add_image('Generated Images', grid, epoch)

        epoch_duration = time.time() - start_time
        writer.add_scalar('Epoch Time Duration', epoch_duration, epoch)
        writer.add_scalar('Average Generator Loss', total_generator_loss / len(dataloader), epoch)
        writer.add_scalar('Average Discriminator Loss', total_discriminator_loss / len(dataloader), epoch)

        if epoch % 50 == 0:  # 每隔50轮次保存一次模型
            os.makedirs('torchgan_model', exist_ok=True)
            torch.save(generator.state_dict(), f'torchgan_model/generator_epoch_{epoch}.pth')
            torch.save(discriminator.state_dict(), f'torchgan_model/discriminator_epoch_{epoch}.pth')

    writer.close()

if __name__ == "__main__":
    main()
