data_mri = '/home/shimy/FusionData/total_mri'
data_pet = '/home/shimy/FusionData/total_pet'
# data_dir = '/home/shimy/FusionData/data_mri'
# model_path = './log/vgg_400.pth'
model_path = 'model/vgg_400.pth'
batch_size = 32
input_size = (227, 227)

class_weight = None

momentum = 0.9
lr = 0.02
epochs = 400

num_print = 20

milestones = [10, 30, 100, 160, 220, 330]
gamma = 0.6
