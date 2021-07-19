import os
import scipy.io
import numpy as np
import nibabel as nib
import shutil

def GiveImageAndTargetLists(main_path):
    images_list = []
    target_list = []
    for folder_name in os.listdir(main_path):
        if folder_name in ['Sandy1','Sandy2']:
            print(folder_name)
            whole_path = main_path + folder_name +"/"
            

            if os.path.isdir(whole_path):
                print(whole_path)
                images_path  = whole_path + "data/"
                targets_path = whole_path + "Lung/"
                
                len_imgs = len([images_path+img for img in os.listdir(images_path) if img.split(".")[-1]=='mat'])
                len_trgs = len([targets_path+trg for trg in os.listdir(targets_path) if trg.split(".")[-1]=='mat'])
                if len_imgs == len_trgs:
                    images_list.append( np.sort([ images_path+img for img in os.listdir(images_path)  if img.split(".")[-1]=='mat']))
                    target_list.append( np.sort([targets_path+trg for trg in os.listdir(targets_path) if trg.split(".")[-1]=='mat']))
                else:

                    imgs_list =[  images_path+img for img in os.listdir(images_path)  if img.split(".")[-1]=='mat' and (img.split(".")[0]+'-lung.mat')   in os.listdir(targets_path)]
                    trg_list = [ targets_path+trg for trg in os.listdir(targets_path) if trg.split(".")[-1]=='mat' and (trg.split(".")[0][:-5] + '.mat') in os.listdir(images_path) ]
                    imgs_list = np.sort(imgs_list)
                    trg_list = np.sort(trg_list)
                    images_list.append(imgs_list)
                    target_list.append(trg_list)

    return (images_list,target_list)



def SavingAsNpy(images_list,target_list,imageTr_path,maskTr_path, prefix=""):

    count = 0
    for i in range(len(images_list)):
        print(i,'/',len(images_list))
        for j in range(len(images_list[i])):#len(images_list[i]
            print(i,' / ',j,'/',len(images_list[i]))
            print(images_list[i][j],'\n',target_list[i][j],'\n')
            #print(count)
            print()
        
            # load img with all channels
            imgs = scipy.io.loadmat(images_list[i][j])
            imgs_array = np.array(imgs['CT'])
            # load the corresponding nodule segmentation
            '''
            if 'tumor' in imgs.keys():
                masks_array = np.array(imgs['tumor'])
            else:
                print('TUMOR1 IS BEING USED')
                masks_array = np.array(imgs['tumor1'])       
            '''
            #load the lung segmentation   
            masks = scipy.io.loadmat(target_list[i][j])
            masks_array_l = np.array(masks['lung'])
        
            #Save 10% of patients data in the test files
              

            image_path  = imageTr_path
            mask_path   = maskTr_path

            dst_img_name = "lu_"+ prefix +'_'+ str(count).zfill(5) + ".nii.gz"
            dst_img_path = os.path.join(image_path, dst_img_name)
            img_nifti = nib.Nifti1Image(imgs_array, np.diag(np.append(imgs['img_resolution'],1.0)))
            img_nifti.to_filename(dst_img_path)

            dst_label_name = "lu_"+ prefix +'_'+ str(count).zfill(5) + ".nii.gz"
            dst_mask_path = os.path.join(mask_path, dst_label_name)
            mask_nifti = nib.Nifti1Image(masks_array_l, np.diag(np.append(imgs['img_resolution'],1.0)))
            mask_nifti.to_filename(dst_mask_path)
            
            count += 1
            
    return (count)



if __name__=='__main__':


    # Destination directory
    nnUnet_im_lbl_folder = "/Data/MoriRichardProject/NiftiLungSeg"
                            

    #if os.path.exists(nnUnet_im_lbl_folder):
        #shutil.rmtree(nnUnet_im_lbl_folder)

    imageTr_path = os.path.join(nnUnet_im_lbl_folder, "images")
    os.makedirs(imageTr_path,exist_ok=True)

    maskTr_path = os.path.join(nnUnet_im_lbl_folder, "Lunglabels")
    os.makedirs(maskTr_path,exist_ok=True)


    
    #Getting images_list & target_list
    raw_dot_m_files = '/Data/Lung/'

    images_list , target_list = GiveImageAndTargetLists(raw_dot_m_files)
    print("images_list) & len(target_list):",len(images_list),'  &  ' ,len(target_list))


    #DO NOT GIVE PREFIX OF "" AND saving_from_number = 0 FOR ALL THE DATA_SETS
    data_set_number = 9
    prefix = "stim"
    saving_from_number = 0

    
    #Loading .m files, saving as nii.gz
    SavingAsNpy(images_list,    target_list,  \
                imageTr_path,   maskTr_path,  \
                prefix=prefix)
    
