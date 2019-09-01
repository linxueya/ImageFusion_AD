# Multimodal Neuroimaging Feature Learning With Multimodal Convolutional Neuro Networks for Diagnosis of Alzheimer’s Disease
## Introduction
！[image](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1567352132330&di=1c3b7381caa9eddedc5c11b83b56d652&imgtype=0&src=http%3A%2F%2F5b0988e595225.cdn.sohucs.com%2Fimages%2F20181218%2F2fa4a33cadea4fd58609f104b31bd788.jpeg)
> MRI从大脑结构上大脑的萎缩进行研究，DTI用于分析大脑微结构水平的水扩散，而PET则利用脑葡萄糖代谢率进行AD分类。这些生物标志物产生互补的信息，即，不同的模式从不同的角度获取疾病信息，从而提高对疾病模式的理解。9
## ImageFusion-AD

### pre-progress
#### Format
  - DCM NIFTI ECAT HRRT
  每种数据格式对应的文件数目不一样。DCM将每个切片单独保存，而其他的数据格式则以一个文件保存整个大脑样本。这样无法读入数据进而输入神经网络。
  - 而对于单一的格式如DCM，对应的切片数量也不一样，在选择top N切片时，每个切片对应的位置可能会有一定的差异。因此我们需要将每个样本归一化到相同的切片数量。
#### 3D reconstruction
  - 将每个样本重构成3D模型，在重采样抽取每个切片。

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
