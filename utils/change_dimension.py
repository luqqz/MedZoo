import os
from sys import path
import numpy as np
import SimpleITK as sitk

path = "/notebooks/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task999_SLK"
gt_path="E:\\Studia\\MGR\\data\\BTCV_Abdomen\\Abdomen\\RawData\\Training\\label"
target_dir = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)

for root, dirs, files in os.walk(gt_path):
    for file in files:
        if ".nii.gz" in file:
            # idx = int(file[4:7])
            # print(idx)
            itk_file = sitk.ReadImage(os.path.join(root, file))
            print(itk_file.GetDirection())
            if idx >= 200 and idx < 500:
                itk_file = sitk.ReadImage(os.path.join(root, file))
                print(itk_file.GetDirection())
                if itk_file.GetDirection() != target_dir:
                    print(os.path.join(root, file))
                    itk_file = sitk.ReadImage(os.path.join(root, file))
                    data = sitk.GetArrayFromImage(itk_file).astype(np.float)
                    print(data.shape)
                    data = data.transpose((2, 1, 0))
                    data = data[::-1, ::-1, :]
                    print(data.shape)
                    ori_origin = itk_file.GetOrigin()
                    new_ori = (ori_origin[2], ori_origin[1], ori_origin[0])
                    ori_spacing = np.array(itk_file.GetSpacing())
                    new_spacing = [ori_spacing[2], ori_spacing[1], ori_spacing[0]]
                    saveITK = sitk.GetImageFromArray(data)
                    saveITK.SetSpacing(new_spacing)
                    saveITK.SetOrigin(new_ori)
                    saveITK.SetDirection(target_dir)
                    sitk.WriteImage(saveITK, os.path.join(root, file))
