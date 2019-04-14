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

## Data Process

We didn't provide the part of data process. If you want to run the code, you need to generate `train_single_3d.json`, `test_single_3d.json`, `train_3d.json`, `test_3d.json` by yourself.

The schemas are list as follows:

``` 
// train_single_3d.json
[
    [the-path-to-train-image/xxx_001.png, label-of-this-img],
    [the-path-to-train-image/xxx_002.png, label-of-this-img],
    [the-path-to-train-image/xxx_003.png, label-of-this-img],
    ......
]
```

``` 
// test_single_3d.json
[
    [the-path-to-test-image/yyy_001.png, label-of-this-img],
    [the-path-to-test-image/yyy_002.png, label-of-this-img],
    [the-path-to-test-image/yyy_003.png, label-of-this-img],
    ......
]
```

``` 
// train_3d.json
[
    [the-path-to-train-image/xxx_, label-of-this-img],
    [the-path-to-train-image/yyy_, label-of-this-img],
    [the-path-to-train-image/zzz_, label-of-this-img],
    ......
]
```

``` 
// test_3d.json
[
    [the-path-to-test-image/xxx_, label-of-this-img],
    [the-path-to-test-image/yyy_, label-of-this-img],
    [the-path-to-test-image/zzz_, label-of-this-img],
    ......
]
```

## Reference

- [Feng Y, Zhang Z, Zhao X, et al. GVCNN: Group-view convolutional neural networks for 3D shape recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 264-272.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf)
- https://github.com/jongchyisu/mvcnn_pytorch

- https://github.com/ace19-dev/gvcnn
- https://github.com/Cadene/pretrained-models.pytorch