import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, scale_aug=False, rot_aug=False, test_mode=False, num_views=12):
        self.data_list = data_list
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path, class_id = self.data_list[idx]
        # Use PIL instead
        imgs = []

        classes = self.classes

        for suffix in classes:
            im = Image.open(path+"obj_whiteshaded_v"+suffix+'.png').convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return class_id, torch.stack(imgs), path


class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, scale_aug=False, rot_aug=False, test_mode=False):
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']

        self.data_list = data_list
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path, class_id = self.data_list[idx]
        # Use PIL instead
        im = Image.open(path).convert('RGB')
        if self.transform:
            im = self.transform(im)

        return class_id, im, path
