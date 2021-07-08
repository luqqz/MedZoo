import os
import numpy
import shutil
import skimage.transform
import SimpleITK

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enums.datatype import DataType
from enums.organ import Organ

NIFTI_extension = '.nii'
_final_data_path = 'data_final/'
_final_data_processed_path = 'data_final_processed/'
_target_spacing = [1.5, .8, .8]

#
# This script will prepare data for training.
# It will create 'data_final' and 'data_final_processed' directories
# 'data_final' contains original data
# 'data_final_processed' contains preprocessed data
# directories structure:
# - data_final[_processed]
#   - {organ}
#     - test
#       - {index}_{organ}_ct.nii[.gz]
#     - train
#       - {index}_{organ}_ct.nii[.gz]
#       - {index}_{organ}_mask.nii[.gz]
#

#
# Supported datasets:
# - LITS
# - KITS
# - Spleen (MSD)
#

def check_and_create_data_dirs(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(os.path.join(data_path, Organ.KIDNEY.value)):
        os.makedirs(os.path.join(data_path, Organ.KIDNEY.value))
        os.makedirs(os.path.join(data_path, Organ.KIDNEY.value, 'train'))
        os.makedirs(os.path.join(data_path, Organ.KIDNEY.value, 'test'))
    if not os.path.exists(os.path.join(data_path, Organ.LIVER.value)):
        os.makedirs(os.path.join(data_path, Organ.LIVER.value))
        os.makedirs(os.path.join(data_path, Organ.LIVER.value, 'train'))
        os.makedirs(os.path.join(data_path, Organ.LIVER.value, 'test'))
    if not os.path.exists(os.path.join(data_path, Organ.SPLEEN.value)):
        os.makedirs(os.path.join(data_path, Organ.SPLEEN.value))
        os.makedirs(os.path.join(data_path, Organ.SPLEEN.value, 'train'))
        os.makedirs(os.path.join(data_path, Organ.SPLEEN.value, 'test'))
    if not os.path.exists(os.path.join(data_path, Organ.BTCV.value)):
        os.makedirs(os.path.join(data_path, Organ.BTCV.value))
        os.makedirs(os.path.join(data_path, Organ.BTCV.value, 'test'))

def rename_and_move(data_root_dir='data/'):
    check_and_create_data_dirs(_final_data_path)

    for root, _, files in os.walk(data_root_dir, topdown=False):
        for file in files:
            if 'LITS' in root:
                if 'volume' in file:
                    index = file[7:-4]
                    shutil.move(os.path.join(root, file), os.path.join(_final_data_path, Organ.LIVER.value, 'train', str(index) + file.replace('volume-' + str(index), '_liver_ct')))
                elif 'segmentation' in file:
                    index = file[13:-4]
                    shutil.move(os.path.join(root, file), os.path.join(_final_data_path, Organ.LIVER.value, 'train', str(index) + file.replace('segmentation-' + str(index), '_liver_mask')))
            elif 'kits' in root:
                index = root[-5:]
                if 'imaging' in file:
                    shutil.move(os.path.join(root, file), os.path.join(_final_data_path, Organ.KIDNEY.value, 'train', str(index) + file.replace('imaging', '_kidney_ct')))
                elif 'segmentation' in file:
                    shutil.move(os.path.join(root, file), os.path.join(_final_data_path, Organ.KIDNEY.value, 'train', str(index) + file.replace('segmentation', '_kidney_mask')))
            elif 'Spleen' in root:
                index = file[7:-7]
                if 'imagesTr' in root:
                    shutil.move(os.path.join(root, file), os.path.join(_final_data_path, Organ.SPLEEN.value, 'train', str(index) + file.replace('spleen_' + str(index), '_spleen_ct')))
                    shutil.move(os.path.join(root, file).replace('imagesTr', 'labelsTr'), os.path.join(_final_data_path, Organ.SPLEEN.value, 'train', str(index) + file.replace('spleen_' + str(index), '_spleen_mask')))
                elif 'imagesTs' in root:
                    shutil.move(os.path.join(root, file), os.path.join(_final_data_path, Organ.SPLEEN.value, 'test', str(index) + file.replace('spleen_' + str(index), '_spleen_ct')))
            elif 'BTCV' in root:
                index = file[3:-7]
                if 'img' in root:
                    shutil.move(os.path.join(root, file), os.path.join(_final_data_path, Organ.BTCV.value, 'test', str(index) + file.replace('img' + str(index), '_btcv_ct')))
                    mask = file.replace('img', 'label')
                    mask_root = root.replace('img', 'label')
                    shutil.move(os.path.join(mask_root, mask), os.path.join(_final_data_path, Organ.BTCV.value, 'test', str(index) + mask.replace('label' + str(index), '_btcv_mask')))

def respace():
    check_and_create_data_dirs(_final_data_processed_path)

    for root, _, files in os.walk(_final_data_path, topdown=False):
        for file in files:
            if NIFTI_extension in file:
                print("Processing: %s" % file)
                file_path = os.path.join(root, file)
                target_file_path = file_path.replace(_final_data_path, _final_data_processed_path)

                if os.path.exists(target_file_path):
                    print("File already exists, continue...")
                    continue

                image = SimpleITK.ReadImage(file_path)
                array_image = SimpleITK.GetArrayFromImage(image)

                size = numpy.array(image.GetSize())[[2, 1, 0]]
                spacing = numpy.array(image.GetSpacing())[[2, 1, 0]]
                origin = image.GetOrigin()
                direction = image.GetDirection()

                if Organ.KIDNEY.value in file:
                    task = Organ.KIDNEY
                    array_image = array_image.transpose((2, 1, 0))
                    size = numpy.array([size[2], size[1], size[0]])
                    spacing = numpy.array([spacing[2], spacing[1], spacing[0]])
                    origin = tuple([origin[2], origin[1], origin[0]])
                    order = [6, 7, 8, 3, 4, 5, 0, 1, 2]
                    old_direction = direction
                    direction = tuple([old_direction[i] for i in order])
                elif Organ.LIVER.value in file:
                    task = Organ.LIVER
                elif Organ.SPLEEN.value in file:
                    task = Organ.SPLEEN
                elif Organ.BTCV.value in file:
                    task = Organ.BTCV
                else:
                    print("Task not supported, continue...")
                    continue

                spacing_ratio = spacing / _target_spacing
                data_type = array_image.dtype

                if DataType.CT.value.lower() in file:
                    data_type = numpy.int32
                    mode = 'constant'
                    order = 3
                else:
                    mode = 'edge'
                    order = 0

                array_image = array_image.astype(numpy.float)
                image_resized = skimage.transform.resize(
                    image=array_image,
                    output_shape=(
                        int(size[0] * spacing_ratio[0]),
                        int(size[1] * spacing_ratio[1]),
                        int(size[2] * spacing_ratio[2])
                    ),
                    order=order,
                    mode=mode,
                    cval=0,
                    clip=True,
                    preserve_range=True
                )
                image_resized = numpy.round(image_resized).astype(data_type)

                target_file = SimpleITK.GetImageFromArray(image_resized)
                target_file.SetSpacing(numpy.array(_target_spacing)[[2, 1, 0]])
                target_file.SetOrigin(origin)
                target_file.SetDirection(direction)

                SimpleITK.WriteImage(target_file, target_file_path)

rename_and_move()
respace()