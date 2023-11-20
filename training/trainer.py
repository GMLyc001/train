# import torch
# import torchvision.utils as vutils
# import logging
# from torchgan.trainer import Trainer
# #import train_config
# from torchgan.losses import MinimaxGeneratorLoss, MinimaxDiscriminatorLoss

# # 创建损失函数实例
# generator_loss = MinimaxGeneratorLoss()
# discriminator_loss = MinimaxDiscriminatorLoss()
# losses = [generator_loss, discriminator_loss]


# class GANTrainer:
#     def __init__(self, network, losses, dataloader, device, batch_size, epochs):
#         # network 参数可能需要为字典形式
#         self.trainer = Trainer(network, losses, sample_size=batch_size, epochs=epochs, device=device)
#         self.dataloader = dataloader

#     def train(self):
#         for epoch in range(self.trainer.epochs):
#             for i, data in enumerate(self.dataloader):
                
#                 # 每个epoch结束后执行
#                 if i == len(self.dataloader) - 1:
#                     # 保存模型
#                     torch.save(self.trainer.generator.state_dict(), f'results/models/generator_epoch_{epoch}.pth')
#                     torch.save(self.trainer.discriminator.state_dict(), f'results/models/discriminator_epoch_{epoch}.pth')
                    
#                     # 保存图像样本
#                     sample_images = self.trainer.sample_generator_output() 
#                     vutils.save_image(sample_images, f'results/images/generated_epoch_{epoch}.png')
                    
#                     # 记录日志
#                     logging.info(f'Epoch {epoch} completed')

#         self.trainer.complete()
