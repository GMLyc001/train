# 训练配置参数
config = {
    "epochs": 2000,              # 训练的总轮次
    "batch_size": 64,            # 每个批次的大小
    "learning_rate_gen": 0.0005, # 生成器的学习率
    "learning_rate_disc": 0.0006,# 判别器的学习率
    "beta1": 0.5,                # Adam优化器的beta1参数
    "beta2": 0.999,              # Adam优化器的beta2参数
    "img_size": 256,             # 图像尺寸
    "channels": 3,               # 图像通道数
    "noise_dim":100
}
