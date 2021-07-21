from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from glob import glob

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, scale=4):
        hr_height, hr_width = hr_shape
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height//scale, hr_width//scale), Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )

        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )
        self.files = glob(root + '/*.*')

    def __getitem__(self, index):
        # TODO : Geometric augmentation

        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"img_lr" : img_lr, "img_hr" : img_hr}

    def __len__(self):
        return len(self.files)

