
import numpy as np
from PIL import Image
import pandas as pd
import os
import numpy as np
import nibabel as nib
import skimage.io as io
import SimpleITK as sitk


def Patients_list(df, col='Pathological.T.Stage' ,categories=[['1a'],['1b'],[3,4]], train_num=24):
    shuffle = True
    whole_D ={}
    for cat in categories:
        list_temp = []     
        for sub_cat in cat:
            list_temp.append([x[-7:] for x in df[df[col] == sub_cat]["PatientID"].values])

        list_temp = np.array([x for y in list_temp for x in y])              # to make a flatten list in case
        key = str(cat[0]) if len(cat)==1 else '{},{}'.format(cat[0],cat[1])  #"['1a']", '[3, 4]' -> '1a', '3,4'
        whole_D[key] = list_temp

    # Shuffle & select
    train_D = {}  
    val_D = {}
    for key in whole_D:
        if shuffle:
            indices = np.arange(len(whole_D[key]))
            np.random.shuffle(indices)
            whole_D[key] = whole_D[key][indices]
        
        train_D[key] = whole_D[key][:train_num]
        val_D[key] = whole_D[key][train_num:]
        
    return train_D, val_D



def GiveImageAndTargetLists(main_path):
    #C:/Users/MSalehjahromi/Data_ICON/
    CT_list = []
    Label_list = []

    CT_path_file  = os.path.join( main_path + "Images_Nifty")
    Label_path_file = os.path.join( main_path + "Lunglabels_Nifty")

    for folder_name in os.listdir(CT_path_file):
        #print(folder_name)
        whole_path_CT = os.path.join(CT_path_file , folder_name )
        CT_list.append(whole_path_CT)

    for folder_name in os.listdir(Label_path_file):
        #print(folder_name)
        whole_path_Label = os.path.join(Label_path_file , folder_name )
        Label_list.append(whole_path_Label)
            
    return (CT_list, Label_list)

'''
def rename(file_list):
    for file in file_list:
        os.rename(file, file[:-12]+file[-7:])
'''


def SavingAsNpy(CT_list, Label_list, CT_Tr_path, Label_Tr_path, CT_Ts_path, Label_Ts_path, CT_Va_path,  Label_Va_path, train_D_list, val_D_list, prefix=""):


    for j in range(len(CT_list)):

        case_name = CT_list[j].split('/')[-1]

        if case_name in train_D_list:

            print( 'Saving case in Train folder:   ', case_name)
            CT = io.imread(CT_list[j], plugin='simpleitk') #CT = np.array(CT)      # for Sandy12   (243, 512, 512)
            Label = io.imread(Label_list[j], plugin='simpleitk') #Label = np.array(Label)    
            CT_path  = CT_Tr_path
            Label_path   = Label_Tr_path

            CT = np.array( CT )
            Label = np.array( Label )

            dst_img_name =  case_name + ".npy"
            dst_img_path = os.path.join(CT_path, dst_img_name)
            np.save(dst_img_path, CT)

            dst_label_name = case_name + ".npy"
            dst_mask_path = os.path.join(Label_path, dst_label_name)
            np.save(dst_mask_path, Label)
            

        elif CT_list[j].split('/')[-1] in val_D_list:

            print( 'Saving case in Val folder:   ', case_name)
            CT = io.imread(CT_list[j], plugin='simpleitk') #CT = np.array(CT)      # for Sandy12   (243, 512, 512)
            Label = io.imread(Label_list[j], plugin='simpleitk') #Label = np.array(Label)    

            CT_path  = CT_Va_path
            Label_path   = Label_Va_path

            CT = np.array( CT )
            Label = np.array( Label )

            dst_img_name =  case_name + ".npy"
            dst_img_path = os.path.join(CT_path, dst_img_name)
            np.save(dst_img_path, CT)

            dst_label_name = case_name + ".npy"
            dst_mask_path = os.path.join(Label_path, dst_label_name)
            np.save(dst_mask_path, Label)



if __name__=='__main__':

    
    col = 'Pathological.T.Stage'
    categories = [['1a'],['1b'],[3,4]]
    train_num = 24

    df = pd.read_excel('Sandy.xlsx')
    train_D, val_D = Patients_list(df, col=col ,categories=categories, train_num=train_num)
    train_D_list = [x for y in train_D.values() for x in y]
    val_D_list = [x for y in val_D.values() for x in y]


    # Destination directory
    main_folder = '/Data/MoriRichardProject/' + col
    os.makedirs(main_folder,exist_ok=True)
    #if os.path.exists(nnUnet_im_lbl_folder):
        #shutil.rmtree(nnUnet_im_lbl_folder)

    # Train
    CT_Tr_path = os.path.join(main_folder, "CT_Tr")
    os.makedirs(CT_Tr_path,exist_ok=True)

    Label_Tr_path = os.path.join(main_folder, "Label_Tr")
    os.makedirs(Label_Tr_path,exist_ok=True)

    # Val
    CT_Va_path = os.path.join(main_folder, "CT_Va")
    os.makedirs(CT_Va_path,exist_ok=True)

    Label_Va_path = os.path.join(main_folder, "Label_Va")
    os.makedirs(Label_Va_path,exist_ok=True)

    # Test
    CT_Ts_path = os.path.join(main_folder, "CT_Ts")
    os.makedirs(CT_Ts_path,exist_ok=True)

    Label_Ts_path = os.path.join(main_folder, "Label_Ts")
    os.makedirs(Label_Ts_path,exist_ok=True)
    

    #Getting images_list & target_list
    NiftiPath = '/Data/MoriRichardProject/NiftiAug2/'
    CT_list, Label_list = GiveImageAndTargetLists(NiftiPath)

    print("len(CT_list) & len(Label_list):",len(CT_list),'  &  ' ,len(Label_list))


    #Let's shuffle them
    indices = np.arange(len(CT_list))
    np.random.shuffle(indices)
    CT_list, Label_list = np.array(CT_list),  np.array(Label_list)     #CT_list, Label_list = np.sort(CT_list),  np.sort(Label_list)
    
    CT_list = CT_list[indices]
    Label_list = Label_list[indices]

    #print(CT_list[:5])
    #print()
    #print(Label_list[:5])

    prefix = ""
    #SavingD
    SavingAsNpy(CT_list,    Label_list,  
                CT_Tr_path,  Label_Tr_path, 
                CT_Ts_path,  Label_Ts_path,
                CT_Va_path,  Label_Va_path,
                train_D_list, val_D_list,
                prefix=prefix)

