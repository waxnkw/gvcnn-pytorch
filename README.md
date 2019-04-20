# GVCNN-Pytorch Version
Environment: python 3.6   PyTorch 1.0.0

The code is implemented on the base of jongchyisu's [mvcnn_pytorch](https://github.com/jongchyisu/mvcnn_pytorch), and is inspired by Sean Kim's [tensorflow version](https://github.com/ace19-dev/gvcnn).

The inception v4 can be found [here](https://github.com/Cadene/pretrained-models.pytorch).

## Paper
![](resources/gvcnn.png)

The paper can be found [here](http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf).

## Modification
There are two main difference from the origin paper:

1. Considering the huge amount of fc layer, we use 1*1 conv instead of fc in group schema module. 
2. If all views' scores are very small, it may cause some problem in params' update. So we add a softmax to normalize the scores generate by group schema.

## Run

modify `root` in data_process.py to your ModelNet40 dataset.

``` sh
python data_process.py
```

```sh
python gvcnn_train.py
```

## Reference

- [Feng Y, Zhang Z, Zhao X, et al. GVCNN: Group-view convolutional neural networks for 3D shape recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 264-272.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf)
- https://github.com/jongchyisu/mvcnn_pytorch

- https://github.com/ace19-dev/gvcnn
- https://github.com/Cadene/pretrained-models.pytorch
