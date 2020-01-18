import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


data1_loss =np.loadtxt("acc_se.txt")
data2_loss = np.loadtxt("acc_cat.txt")
data3_loss = np.loadtxt("acc_res.txt")
data4_loss = np.loadtxt("acc_sig.txt")

x1 = range(len(data1_loss))
y1 = data1_loss
x2 = range(len(data2_loss))
y2 = data2_loss
x3 = range(len(data3_loss))
y3 = data3_loss
x4 = range(len(data4_loss))
y4 = data4_loss

y1 = savgol_filter(y1,31,3)
y2 = savgol_filter(y2,31,3)
y3 = savgol_filter(y3,31,3)
y4 = savgol_filter(y4,31,3)



fig = plt.figure(figsize = (7,5))       #figsize是图片的大小`
ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`

pl.plot(x1,y1,'r-',label=u'DMFNet(fusion+extraction)')
# ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
p2 = pl.plot(x2, y2,'g-', label = u'DMFNet(fusion)')
p3 = pl.plot(x3, y3,'b-', label = u'ResNet_com')
p3 = pl.plot(x4, y4,'k-', label = u'ResNet')
pl.legend()
#显示图例

pl.xlabel(u'iters')
pl.ylabel(u'accuracy')
plt.title('Compare accuracy for different models in training')
plt.savefig('acc.jpg')
