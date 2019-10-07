# data_dir = '/home/shimy/PythonProjects/ad-resnet-50/MRI_32_Images/data_00'
data_dir = '/home/shimy/FusionData/data_mri'
# model_path = './log/vgg_400.pth'
model_path ='./model/vggcm2_300.pth'
batch_size = 32
input_size = (160, 160)

class_weight = None

momentum = 0.9
lr = 0.002
epochs = 300

num_batch_print = 80

milestones = [10, 30, 100, 160, 220, 330]
gamma = 0.5
