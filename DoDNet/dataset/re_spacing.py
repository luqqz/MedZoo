import numpy as np
import os
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk

spacing = {
    0: [1.5, 0.8, 0.8],
    1: [3, 0.76, 0.76],
    2: [1.5, 0.8, 0.8],
    3: [1.5, 0.8, 0.8],
    4: [1.5, 0.8, 0.8],
    5: [1.5, 0.8, 0.8],
    6: [1.5, 0.8, 0.8],
    7: [1.5, 0.8, 0.8],
}

ori_path = './test'
new_path = './0123456_spacing_same'

count = -1
for root1, dirs1, _ in os.walk(ori_path):
    for i_dirs1 in tqdm(sorted(dirs1)):  # 0Liver
        #############################################################################
        for root2, dirs2, files2 in os.walk(os.path.join(root1, i_dirs1)):
            for i_dirs2 in sorted(dirs2):  # imagesTr

                for root3, dirs3, files3 in os.walk(os.path.join(root2, i_dirs2)):
                    for i_files3 in sorted(files3):
                        if i_files3[0] == '.':
                            continue
                        save_path = os.path.join(new_path, i_dirs1, i_dirs2)
                        #if os.path.exists(os.path.join(save_path, i_files3)):
                        #    print("File already exists, continue...")
                        #    continue
                        # read img
                        print("Processing %s" % (i_files3))
                        img_path = os.path.join(root3, i_files3)
                        imageITK = sitk.ReadImage(img_path)
                        image = sitk.GetArrayFromImage(imageITK)
                        ori_size = np.array(imageITK.GetSize())[[2, 1, 0]]
                        ori_spacing = np.array(imageITK.GetSpacing())[[2, 1, 0]]
                        ori_origin = imageITK.GetOrigin()
                        ori_direction = imageITK.GetDirection()

                        task_id = int(i_dirs1[0])
                        target_spacing = np.array(spacing[task_id])
                        spc_ratio = ori_spacing / target_spacing

                        data_type = image.dtype
                        if i_dirs2 != 'labelsTr':
                            data_type = np.int32

                        if i_dirs2 == 'labelsTr':
                            order = 0
                            mode_ = 'edge'
                        else:
                            order = 3
                            mode_ = 'constant'

                        image = image.astype(np.float)

                        image_resize = resize(image, (int(ori_size[0] * spc_ratio[0]), int(ori_size[1] * spc_ratio[1]),
                                                      int(ori_size[2] * spc_ratio[2])),
                                              order=order, mode=mode_, cval=0, clip=True, preserve_range=True)
                        image_resize = np.round(image_resize).astype(data_type)

                        # save

                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        saveITK = sitk.GetImageFromArray(image_resize)
                        saveITK.SetSpacing(target_spacing[[2, 1, 0]])
                        saveITK.SetOrigin(ori_origin)
                        saveITK.SetDirection(ori_direction)
                        if i_dirs2 == "imagesTr":
                            sitk.WriteImage(saveITK, os.path.join(save_path, i_files3.replace("img00", "img10")))
                        else:
                            sitk.WriteImage(saveITK, os.path.join(save_path, i_files3.replace("label00", "label10")))