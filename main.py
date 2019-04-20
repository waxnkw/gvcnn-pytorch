from tools.MvcnnDataset import MultiviewImgDataset
import json

if __name__ == '__main__':
    train_file = open("./train_3d.json")
    train_list = json.load(train_file)
    d = MultiviewImgDataset(train_list)
    class_id, imgs, path = d.__getitem__(0)
    print(class_id)
    print(path)
    print(imgs.shape)
