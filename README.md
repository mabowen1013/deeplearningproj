目前我做了四次训练
- 第一次是不信邪用完整数据，使用EfficientNet-B0基础模型训练，练太慢中途停了。
- 第二次是对数据进行subsample后，用基础模型进行了50 epoch训练，val的accuracy在72%左右
- 第三次就是在EfficientNet-B0基础上design了新的MoE variant
    - 用MoE layer替换了EfficientNet-B0的stage5, stage6, stage7的1x1 conv，然后再stage6和stage 7之间加入了一个MoE FFN block
    - 然后这个模型练了大概33个epoch，val的accuracy在60左右就不动了。
- 然后第四次修改design方案
    - stage5不改了
    - MoE替换stage6和stage7的1x1 conv
    - 拉高了一点参数量
    - MoE FFN block依旧加在 stage6和stage7之间。
    - 这个模型练了40多epoch，val的accuracy在70%，和基础模型的准确度非常接近，并且激活的参数比基础模型少了20%，所以结果还是非常不错的。
- 然后模型存在checkpoints文件夹里

主要的代码都是在model文件夹下。
