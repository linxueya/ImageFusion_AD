import  os
import glob
print(os.getcwd())  #获取当前工作目录路径

root_path = '/home/shimy/ADNI/NC/ADNI'
imgs = os.listdir(root_path)
img_num=[]
for img in imgs:
    img1 = os.path.join(root_path,img)
    img2 = os.listdir(img1)
    img3 = os.path.join(img1,img2[0])
    img4 = os.listdir(img3)
    img5 = os.path.join(img3,img4[0])
    img6 = os.listdir(img5)
    img7 = os.path.join(img5,img6[0])
    os.chdir(img7)
    n=len(os.listdir(img7))
    img_num.append(n)
    print(img,n)


result_dic={}
for item_str in img_num:
    if item_str not in result_dic:
        result_dic[item_str]=1
    else:
        result_dic[item_str]+=1
print(result_dic)