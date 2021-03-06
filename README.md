

## [Multi-modal Neuroimaging Feature Fusion for Diagnosis of Alzheimer’s Disease](https://doi.org/10.1016/j.jneumeth.2020.108795)

> [TaoZhang](https://www.sciencedirect.com/science/article/pii/S0165027020302181?via%3Dihub#!) [MingyangShi](https://www.sciencedirect.com/science/article/pii/S0165027020302181?via%3Dihub#!) 
>
> School of Electronic and Information Engineering, Tianjin University, 300387, Tianjin, China

## Introduction
> Compared to single-modal neuroimages classification of AD, multi-modal classification can achieve better performance by fusing different information. Exploring synergy among various multi-modal neuroimages is contributed to identifying the pathological process of neurological disorders. However, it is still problematic to effectively exploit multi-modal information since lack of effective fusion method. In this paper, we propose a deep multi-modal fusion network based on the attention mechanism, which can selectively extract features from MRI and PET branches and suppress the irrelevant information. In the attention model, the ratio of each modal fusion is assigned automatically according to the importance of the data. Different from the early fusion and the late fusion, a hierarchical fusion method is adopted to ensure that original features are not damaged in feature fusion. Benefit from the attention model with hierarchical structure, the proposed network is capable of exploiting low-level and high-level features. Evaluating the model on the ADNI dataset, the experimental results show that it outperforms the state-of-the-art methods. In particular, the final classification results of the NC/AD, SMCI/PMCI and Four-Class are 95.21 %, 89.79%, and 86.15%, respectively.
## Implement Details

### pre-progress
#### Format
  - DCM NIFTI ECAT HRRT
    每种数据格式对应的文件数目不一样。DCM将每个切片单独保存，而其他的数据格式则以一个文件保存整个大脑样本。这样无法读入数据进而输入神经网络。
  - 而对于单一的格式如DCM，对应的切片数量也不一样，在选择top N切片时，每个切片对应的位置可能会有一定的差异。因此我们需要将每个样本归一化到相同的切片数量。
  - 综合上述因素最后选择的是NIFTI
#### 3D reconstruction
  - 将每个样本重构成3D模型，在重采样抽取每个切片。
#### Toolbox
- spm:fmri处理需要slice timing， pet不需要
  pre_fix a  slice timing 隔层扫描（1 3 5 ……6 4 2）消除时间上的差异，线性回归
- ​        r  realign 头动校正
  ​        y  coregister 在这一步需要AC校正（display设置原点）
  ​        c  segment 分割结构像（白质，灰质，脑脊液）
  ​        w  normalise
  ​        s  smooth
- cat12 ：- mwp1outputResult gray mater 
  - mwp2outputResult white mater
    p0outputResult   unknown  
    wmoutputResult   unknown  


### discussion
- MULTIMODAL REPRESENTATIONS:
> 多模态表示是使用来自多个此类实体的信息表示数据。表示多种模式带来了许多困难:如何组合来自不同来源的数据;如何处理不同程度的噪音;以及如何处理丢失的数据。以有意义的方式表示数据的能力对于多模态问题是至关重要的，并且是任何模型的基础。
>- Joint Representations
>联合表示的最简单的例子是单个模态特征的串联(也称为早期融合)。在本节中，我们将讨论创建联合表示的更高级方法，首先从神经网络开始，然后是图形模型和递归神经网络
>- Neural networks
>由于深层神经网络的多层特性，假设每一层后续层都以更抽象的方式表示数据，因此通常使用最后一层或倒数第二层作为数据表示形式
>- 联合多模态表示本身通过多个隐层传递或直接用于预测。这样的模型可以端到端训练——学习表示数据和执行特定任务。这使得神经网络的多模态表示学习与多模态融合有着密切的关系
>- Probabilistic graphical models
> 多模态DBMs能够通过合并两个或多个无向图，并在它们之上使用一个隐藏单元的二进制层，从多个模态学习联合表示。由于模型的无方向性，它们允许每种模式的低水平表示在联合训练之后相互影响。
> 使用多模态DBMs学习多模态表示的一大优点是它们的生成特性，这使得处理丢失数据的方法很简单——即使丢失了整个模态，模型也有一种自然的处理方法。


- FUSION：
>对多模态融合的兴趣来自于它能提供的三个主要好处。首先，获得观察同一现象的多种模式可能使预测更加可靠。AVSR社区对此进行了特别的探索和利用[163]。第二，获得多种模式可能使我们能够获得互补的信息，这在个别模式中是不可见的。第三，当其中一种模式缺失时，多模态系统仍然可以运行，例如当人不说话时，从视觉信号中识别情绪。
>- Model-agnostic approaches
>早期的融合在特征提取之后立即集成它们(通常通过简单地连接它们的表示)。另一方面，晚期融合在每种模式做出决定(例如分类或回归)之后执行集成。与模型无关的方法的优点是，它们几乎可以使用任何单模分类器或回归器来实现。然而，晚期融合忽略了模式间低水平的相互作用。
>- Model-based approaches
> 由于内核可以看作是数据点之间的相似函数，所以MKL中特定于模式的内核可以更好地融合异构数据。
>- Neural Networks
> 深度神经网络方法在数据融合中的一大优点是能够从大量的数据中学习。其次，最近的神经结构允许对多模态表示组件和融合组件进行端到端训练。最后，与非神经网络系统相比，该方法具有较好的性能，能够学习其他方法难以学习的复杂决策边界.缺点是是它们缺乏可解释性。很难判断预测的依据是什么，以及哪种模式或特征发挥了重要作用。此外，神经网络需要大量的训练数据集才能成功。

- FUSION GOAL：
>- 本课题拟设计一种端到端的多模态数据生成——判别模型，多模态的数据融合采用。生成模型综合DBM和神经网络的优缺点，并在其中寻找突破点。当有多个模态被用于训练判别模型时，一部分的单模态输入被屏蔽。即使没有一种模式，隐藏的神经元也可以被训练并对结果做出预测。该策略的目的是使第一个隐藏层的一些隐藏神经元容易被独立的模态激活。即使是只有单模态的数据，也能应用于被训练的网络。

- 对原始图像进行融合；这涉及图像的配准以及对齐，SPM软件可以完成这些预处理并对图像进行融合。
- 对特征图进行融合；之前看到有论文提取图像SIFT和KAZE特征，然后再进行分类，我们也可以基于此提取融合再分类。另外就是可以尝试融合卷积之后的特征图。（这方面还不是很了解）
- 用两个网络分别训练MRI,PET。在SOFTMAX之前将两个网络串联起来。
![image](https://i.loli.net/2018/12/18/5c18dc7cc6c10.png)

### 神经网络训练
> 利用deep-fused nets论文里的思想对两种模态的网络进行融合，主要分为3个子网络，2个深层网络负责提取两种模态的特征，一个浅层网络负责融合

- dataloader的均值方差选择

```
if 'coco' in args.dataset:
   mean_vals = [0.471, 0.448, 0.408]
   std_vals = [0.234, 0.239, 0.242]
elif 'imagenet' in args.dataset:
   mean_vals = [0.485, 0.456, 0.406]
   std_vals = [0.229, 0.224, 0.225]
```
- 遇到的问题
> cat12 的输出文件是什么？
  mwp1outputResult gray mater 
> mwp2outputResult white mater
  p0outputResult   unknown  
  wmoutputResult   unknown  

### deep fused net
> 以deep fused net 和 acnet 为例对多模态融合网络进行优化
    

### 
