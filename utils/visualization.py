import matplotlib.pyplot as plt
import nibabel
import imageio
import numpy
from skimage.transform import resize
import matplotlib.colors as colors
import os

path_spleen_2d = "D:\Studia\MGR\RESULTS\out_spl_2d\output_spleen_2d"
path_spleen_3d = "D:\Studia\MGR\RESULTS\out_spl_3d3\output_spl_33D2"

path_all_2d = "D:\Studia\MGR\RESULTS\out_all_2d\output_all_2d"
path_all_3d = "D:\Studia\MGR\RESULTS\output_all_3d2\output_all_3d2"
path_all_dodnet = "D:\Studia\MGR\RESULTS\inference-dodnet\my" # 0TEST, 1TEST, 2TEST
path_all_pipofan = "D:\Studia\MGR\RESULTS\pred-chybaPIPO\pred"

pipofan_name = "pred"
dodnet_name_prefix = "label"
dodnet_name_sufix = "_pred"

sample_index = 5

output_dod = "D:\Studia\MGR\RESULTS\PRESENTATION\\5\dod7"
output_pip = "D:\Studia\MGR\RESULTS\PRESENTATION\\5\pipo7"
output_2d = "D:\Studia\MGR\RESULTS\PRESENTATION\\5\\2d7"
output_3d = "D:\Studia\MGR\RESULTS\PRESENTATION\\5\\3d7"
output_all = "D:\Studia\MGR\RESULTS\PRESENTATION\\5\\all"

original_image_path = "D:\Studia\MGR\data\BTCV_Abdomen\Abdomen\RawData\Training\img"
original_mask_path = "D:\Studia\MGR\data\BTCV_Abdomen\Abdomen\RawData\Training\label"
original_image = nibabel.load(os.path.join(original_image_path, "img" + str(sample_index).zfill(4) + ".nii.gz"))
original_shape = original_image.shape
target_shape = (original_shape[0] // 2, original_shape[1] // 2, original_shape[2] // 2)
print(target_shape)

# concatenate dodnet_output
dodnet_file0 = nibabel.load(os.path.join(path_all_dodnet, "0TEST", dodnet_name_prefix + str(sample_index).zfill(4) + dodnet_name_sufix + ".nii.gz")).get_fdata()
dodnet_file1 = nibabel.load(os.path.join(path_all_dodnet, "1TEST", dodnet_name_prefix + str(sample_index).zfill(4) + dodnet_name_sufix + ".nii.gz")).get_fdata()
dodnet_file2 = nibabel.load(os.path.join(path_all_dodnet, "2TEST", dodnet_name_prefix + str(sample_index).zfill(4) + dodnet_name_sufix + ".nii.gz")).get_fdata()
dodnet_file = numpy.add(dodnet_file0, dodnet_file1)
dodnet_file = numpy.add(dodnet_file, dodnet_file2)
dodnet_file[dodnet_file > 1] = 1

bounds = numpy.array([-0.5, 0.5, 1.5, 2.5, 3.5])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

indices = []
for i in range(1, target_shape[2]):
    indices.append(str(i).zfill(4))

### ORIGINAL IMAGE AND MASK ###
original_image = resize(original_image.get_fdata(), target_shape)
original_mask = nibabel.load(os.path.join(original_mask_path, "label" + str(sample_index).zfill(4) + ".nii.gz")).get_fdata()
original_mask[original_mask == 1] = 20
original_mask[original_mask == 2] = 21
original_mask[original_mask == 3] = 21
original_mask[original_mask == 6] = 22

original_mask[original_mask < 20] = 0

original_mask[original_mask == 20] = 3
original_mask[original_mask == 21] = 2
original_mask[original_mask == 22] = 1
original_mask = resize(original_mask, target_shape)

idx = 0

for i in indices:
    b = plt.subplot(2, 1, 1)
    b.title.set_text('Original Image')
    a = plt.imshow(original_image[:, :, idx])
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)
    
    b = plt.subplot(2, 1, 2)
    b.title.set_text('Original Mask')
    a = plt.imshow(original_mask[:, :, idx], norm=norm, cmap='RdBu_r')
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)

    plt.savefig(os.path.join(output_all, "zzz" + i + ".png"))
    idx += 1

# Build GIF
with imageio.get_writer(os.path.join(output_all, "zzz" + str(i).zfill(4) + '.gif'), mode='I') as writer:
    for filename in indices:
        image = imageio.imread(os.path.join(output_all, "zzz" + filename + '.png'))
        writer.append_data(image)

### ALL ###
dodnet_file = resize(dodnet_file, target_shape)
pipo_file = nibabel.load(os.path.join(path_all_pipofan, pipofan_name + str(sample_index).zfill(4) + ".nii.gz")).get_fdata()
pipo_file = resize(pipo_file, target_shape)
twod_file = nibabel.load(os.path.join(path_all_2d, "img" + str(sample_index).zfill(4) + ".nii.gz")).get_fdata()
twod_file[twod_file == 1] = 4
twod_file[twod_file == 2] = 5
twod_file[twod_file == 3] = 6
twod_file[twod_file == 4] = 3
twod_file[twod_file == 5] = 1
twod_file[twod_file == 6] = 2
twod_file = resize(twod_file, target_shape)
three_file = nibabel.load(os.path.join(path_all_3d, "img" + str(sample_index).zfill(4) + ".nii.gz")).get_fdata()
three_file[three_file == 1] = 4
three_file[three_file == 2] = 5
three_file[three_file == 3] = 6
three_file[three_file == 4] = 3
three_file[three_file == 5] = 1
three_file[three_file == 6] = 2
three_file = resize(three_file, target_shape)
idx = 0

for i in indices:
    b = plt.subplot(2, 2, 1)
    b.title.set_text('UNet')
    a = plt.imshow(twod_file[:, :, idx], norm=norm, cmap='RdBu_r')
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)
    
    b = plt.subplot(2, 2, 2)
    b.title.set_text('3D UNet')
    a = plt.imshow(three_file[:, :, idx], norm=norm, cmap='RdBu_r')
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)

    b = plt.subplot(2, 2, 3)
    b.title.set_text('PIPO-FAN')
    a = plt.imshow(pipo_file[:, :, idx], norm=norm, cmap='RdBu_r')
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)

    b = plt.subplot(2, 2, 4)
    b.title.set_text('DoDNet')
    a = plt.imshow(dodnet_file[:, :, idx], norm=norm, cmap='RdBu_r')
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)
    plt.savefig(os.path.join(output_all, i + ".png"))
    idx += 1

# Build GIF
with imageio.get_writer(os.path.join(output_all, str(i).zfill(4) + '.gif'), mode='I') as writer:
    for filename in indices:
        image = imageio.imread(os.path.join(output_all, filename + '.png'))
        writer.append_data(image)

# ### DODNET ###
# dodnet_file = resize(dodnet_file, target_shape)
# idx = 0
# for i in indices:
#     plt.imshow(dodnet_file[:, :, idx], norm=norm, cmap='RdBu_r')
#     plt.savefig(os.path.join(output_dod, i + ".png"))
#     idx += 1

# # Build GIF
# with imageio.get_writer(os.path.join(output_dod, str(i).zfill(4) + '.gif'), mode='I') as writer:
#     for filename in indices:
#         image = imageio.imread(os.path.join(output_dod, filename + '.png'))
#         writer.append_data(image)

# ### PIPO-FAN ###
# pipo_file = nibabel.load(os.path.join(path_all_pipofan, pipofan_name + str(sample_index).zfill(4) + ".nii.gz")).get_fdata()
# pipo_file = resize(pipo_file, target_shape)
# idx = 0
# for i in indices:
#     plt.imshow(pipo_file[:, :, idx], norm=norm, cmap='RdBu_r')
#     plt.savefig(os.path.join(output_pip, i + ".png"))
#     idx += 1

# # Build GIF
# with imageio.get_writer(os.path.join(output_pip, str(i).zfill(4) + '.gif'), mode='I') as writer:
#     for filename in indices:
#         image = imageio.imread(os.path.join(output_pip, filename + '.png'))
#         writer.append_data(image)

# ### 2D ###
# twod_file = nibabel.load(os.path.join(path_all_2d, "img" + str(sample_index).zfill(4) + ".nii.gz")).get_fdata()
# twod_file = resize(twod_file, target_shape)
# twod_file[twod_file == 1] = 4
# twod_file[twod_file == 2] = 5
# twod_file[twod_file == 3] = 6
# twod_file[twod_file == 4] = 2
# twod_file[twod_file == 5] = 1
# twod_file[twod_file == 6] = 3

# idx = 0
# for i in indices:
#     plt.imshow(twod_file[:, :, idx], norm=norm, cmap='RdBu_r')
#     plt.savefig(os.path.join(output_2d, i + ".png"))
#     idx += 1

# # Build GIF
# with imageio.get_writer(os.path.join(output_2d, str(i).zfill(4) + '.gif'), mode='I') as writer:
#     for filename in indices:
#         image = imageio.imread(os.path.join(output_2d, filename + '.png'))
#         writer.append_data(image)

# ### 3D ###
# three_file = nibabel.load(os.path.join(path_all_3d, "img" + str(sample_index).zfill(4) + ".nii.gz")).get_fdata()
# three_file[three_file == 1] = 4
# three_file[three_file == 2] = 5
# three_file[three_file == 3] = 6
# three_file[three_file == 4] = 2
# three_file[three_file == 5] = 1
# three_file[three_file == 6] = 3
# three_file = resize(three_file, target_shape)

# idx = 0
# for i in indices:
#     plt.imshow(three_file[:, :, idx], norm=norm, cmap='RdBu_r')
#     plt.savefig(os.path.join(output_3d, i + ".png"))
#     idx += 1

# # Build GIF
# with imageio.get_writer(os.path.join(output_3d, str(i).zfill(4) + '.gif'), mode='I') as writer:
#     for filename in indices:
#         image = imageio.imread(os.path.join(output_3d, filename + '.png'))
#         writer.append_data(image)