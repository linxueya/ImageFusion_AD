# ImageFusion-AD

### pre-progress
#### Format
  DCM NIFTI ECAT HRRT
  - 每种数据格式对应的文件数目不一样。DCM将每个切片单独保存，而其他的数据格式则以一个文件保存整个大脑样本。这样无法读入数据进而输入神经网络。
  - 而对于单一的格式如DCM，对应的切片数量也不一样，在选择top N切片时，每个切片对应的位置可能会有一定的差异。因此我们需要将每个样本归一化到相同的切片数量。
#### 3D reconstruction
  - 

### discussion
1 fusion：
