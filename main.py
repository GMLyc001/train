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
    dataset = ImagesDataset(images_path=r"D:\Anaconda\envs\test\TRYTRY\datasets\picture384\*.jpg")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    '''
     使用PyTorch的DataLoader类创建一个用于迭代训练数据的数据加载器
     shuffle=True: 这表示在每个epoch开始时是否对数据进行随机洗牌。
     将其设置为True可以确保每个epoch中每个批次的数据顺序都是随机的,
     有助于模型学习更泛化的特征
    '''

    generator = Generator()
    discriminator = Discriminator()

    #设置初始画ADM优化器
    optimizer_gen = Adam(generator.parameters(), lr=config['learning_rate_gen'], betas=(config['beta1'], config['beta2']))
    optimizer_disc = Adam(discriminator.parameters(), lr=config['learning_rate_disc'], betas=(config['beta1'], config['beta2']))
    '''
    Adam: 这是使用的优化算法，Adam是一种常用于深度学习的优化算法，它结合了动量（Momentum）和RMSprop的优点，以有效地更新网络权重。
    generator.parameters(): 生成器网络中需要优化的参数（如权重和偏差）。
    lr=config['learning_rate_gen']: 这里设置了学习率（lr），它决定了在每次网络权重更新时参数改变的幅度。
    betas=(config['beta1'], config['beta2']): Adam优化器特有的两个超参数，用于计算梯度的一阶矩估计和二阶矩估计，通常设置为接近1的值。
    '''

    #损失函数
    generator_loss = MinimaxGeneratorLoss()
    discriminator_loss = MinimaxDiscriminatorLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA is available:", torch.cuda.is_available())
    # device = torch.device("cuda:0")
    #device = torch.device("cpu")
    '''
    当 GPU可用时,device 会被设置为 "cuda:0"（即使用第一个 GPU),如果不可用，则使用 CPU。
    '''
    generator.to(device)
    discriminator.to(device)

    writer = SummaryWriter('./torchgan_logs')

    for epoch in range(config['epochs']):
        start_time = time.time()#计算每个epoch的运行时间，以监控训练过程的效率
        total_generator_loss = 0.0#损失植
        total_discriminator_loss = 0.0

        for i,(real_images, labels) in enumerate(dataloader):

            #real_images = data
            real_images = real_images.to(device)#跑GPU上 .to移动
            labels = labels.to(device)
         
            if i == 0 and epoch % 10 == 0:  # 根据需要调整间隔
                noise_dim = config['noise_dim']
                batch_size = config['batch_size']
                noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)  # 创建噪声张量
                fake_images = generator(noise)  # 生成假图像（噪声张量
                grid = vutils.make_grid(fake_images, padding=2, normalize=True)
                writer.add_image('Generated Images', grid, epoch)#日志
                fake_images = generator(noise)

            # 清除之前的梯度
            optimizer_gen.zero_grad()
            optimizer_disc.zero_grad()
            '''
            在PyTorch中，梯度是累加的，这意味着在每次进行反向传播之前，需要手动将梯度归零。
            如果不这么做，梯度将会不断累加，导致训练过程中出现错误的方向或步长，进而影响模型的学习效率和最终性能。
            '''

            # 更新权重
            optimizer_gen.step()
            optimizer_disc.step()
            '''
            在神经网络训练中，一旦完成了一次前向传播和随后的反向传播（计算了损失并通过反向传播得到梯度）
            就需要根据这些梯度来更新网络的权重。
            '''

            fake_images = generator(noise)  # 生成假图像
            real_output = discriminator(real_images)  # 前向传播，通过判别器传递真实图像
            fake_output = discriminator(fake_images)  # 前向传播，通过判别器传递假图像
            # 计算真实图像的损失，判别器对真实图像的输出应被判别为真（标签为1）
            real_loss = discriminator_loss(real_output, torch.ones_like(real_output))
            # 计算假图像的损失，判别器对假图像的输出应被判别为假（标签为0）
            fake_loss = discriminator_loss(fake_output, torch.zeros_like(fake_output))
            # 计算总损失
            total_loss = real_loss + fake_loss

            #累加损失以便后续计算平均损失
            fake_output = discriminator(fake_images)
            g_loss = generator_loss(fake_output)
            total_generator_loss += g_loss.item()

            real_output = discriminator(real_images)
            r_loss = generator_loss(real_output)
            total_generator_loss += r_loss.item()

            # 反向传播和优化
            total_loss.backward()  # 反向传播计算梯度
            optimizer_gen.step()  # 更新生成器的权重
            optimizer_disc.step()  # 更新判别器的权重

            epoch_duration = time.time() - start_time
            writer.add_scalar('Epoch Time Duration', epoch_duration, epoch)

            #累加损失以便后续计算平均损失
            #total_generator_loss += generator_loss.item()
            #total_discriminator_loss += discriminator_loss.item()


            # 计算并记录平均损失
            avg_generator_loss = total_generator_loss / len(dataloader)
            avg_discriminator_loss = total_discriminator_loss / len(dataloader)
            writer.add_scalar('Average Generator Loss', avg_generator_loss, epoch)
            writer.add_scalar('Average Discriminator Loss', avg_discriminator_loss, epoch) 

        if epoch % 50 == 0:  # 每隔50轮次保存一次模型
            os.makedirs('torchgan_model', exist_ok=True)
            '''
            创建一个名为 torchgan_model，用于存放保存的模型文件。
            exist_ok=True 如果目录已经存在，函数不会抛出错误。 
            '''
            torch.save(generator.state_dict(), f'torchgan_model/generator_epoch_{epoch}.pth')#获取模型的参数并保存
            torch.save(discriminator.state_dict(), f'torchgan_model/discriminator_epoch_{epoch}.pth')
 
 

    writer.close()

def calculate_gradients(model):
    total_gradients = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
    mean_gradients = total_gradients / len([p for p in model.parameters() if p.grad is not None])
    return mean_gradients
'''
model.parameters() 返回模型中所有的参数
对于每个有梯度的参数(p.grad is not None)
计算其梯度的范数(p.grad.norm())
范数是梯度向量的大小或长度的度量，常用于评估梯度的大小
sum得到模型中所有参数梯度大小的总和
len获取列表的长度,即有梯度的参数总数
mean_gradients 通过将总梯度 total_gradients 除以有梯度的参数数量来计算得到，这就给出了模型参数梯度的平均大小

'''

if __name__ == "__main__":
    main()
