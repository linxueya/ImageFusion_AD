# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    env = 'default'  # visdom 环境
    vis_port =8097 # visdom 端口
    model = 'AlexNetCom'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = '/home/shimy/FusionData/rank_mri/train'  # 训练集存放路径
    test_data_root = '/home/shimy/FusionData/rank_mri/validation'  # 测试集存放路径
    train_data_root1 = '/home/shimy/FusionData/rank_pet/train'  # 训练集存放路径
    test_data_root1 = '/home/shimy/FusionData/rank_pet/validation'  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    label_name = 'ADNCFUS'  #保存训练模型时使用，以免模型混淆

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    gpu = 'cuda:1'
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = './debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 400
    lr = 0.002  # initial learning rate
    lr_decay = 0.99  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数


    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device =t.device(opt.gpu) if opt.use_gpu else t.device('cpu')


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
