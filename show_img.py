##
import SimpleITK as sitk
import skimage.io as io
import matplotlib.pyplot as plt
def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

# 显示一个系列图
def show_img(data):
    io.imshow(data, cmap='gray')
    io.show()
    # for i in range(data.shape[0]):
    #     io.imshow(data[i, :, :], cmap='gray')
    #     print(i)
    #     io.show()

# 单张显示
def show_img(ori_img):
    io.imshow(ori_img[0], cmap='gray')
    io.show()


path = 'C:/Users/MBSaad/Anaconda3/envs/recon-env2/PROJECTS/Debugging_public_dataset/Prediction/Target/Slice_1995.nii.gz'
path_ori = 'C:/Users/MBSaad/Anaconda3/envs/recon-env2/PROJECTS/Debugging_public_dataset/Prediction/Target/Slice_1995.nii.gz'
#path = '/home/wangchao/Codes/CT_encoder/Prediction/Target/Slice_025.nii.gz'
# path = '/home/wangchao/Codes/CT_encoder/Lung/Test/Target/Slice_030.nii.gz'
data = read_img(path)
data_ori = read_img(path_ori)

# data = data[0, :, :]
#plt.imshow(data)
#plt.show()
#show_img(data)

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(data_ori, cmap='gray')
ax1.set_title('Original')

ax2 = fig.add_subplot(1,2,2)
ax2.imshow(data, cmap='gray')
ax2.set_title('Recon')


