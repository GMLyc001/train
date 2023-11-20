from email import generator
import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
# from generator import Generator
# from discriminator import Discriminator
# from training.trainer import GANTrainer
from training.train_config import config
from torch.utils.data import DataLoader

img_size = 256

class ImagesDataset(data.Dataset):
    def __init__(self, images_path: str):
        self.files = glob.glob(images_path)#使用 glob.glob 函数根据提供的路径模式（images_path）搜索匹配的文件
        # print(self.files)
        self.images = [None] * len(self.files)#初始化长度与 self.files相同的列表，出生在为None,存储处理好的图像

    def __len__(self):
        return len(self.files)#返回数据集

    def __getitem__(self, index):#索引访问数据集元素
        if self.images[index] is None:
            self.images[index] = self.generate_image(index)#未处理则调用generate_image处理
        return self.images[index]

    def generate_image(self, index):#转换序列
        img = Image.open(self.files[index]).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(img_size + img_size // 2),#调大小
            transforms.RandomCrop(img_size),#随机剪裁
            transforms.RandomHorizontalFlip(),#随机反转（水平
            transforms.ToTensor()#转换为张量
        ])
        resized = transform(img)#将所有转换应用到加载的图像上
        return resized, index
 
 
 