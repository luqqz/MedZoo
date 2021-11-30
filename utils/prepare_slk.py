import os
import shutil
import nibabel
import numpy

output_path="../nnUNet/raw/nnUNet_raw_data/Task999_SLK"
lits_input_path="LITS_Training_Batch1/media/nas/01_Datasets/CT/LITS/Training Batch 1/"
kits_input_path="kits19/data/"
spleen_input_path="Task09_Spleen/Task09_Spleen"

if not os.path.exists(output_path):
    os.mkdir(output_path)
    os.mkdir(os.path.join(output_path, "imagesTr"))
    os.mkdir(os.path.join(output_path, "labelsTr"))

for root, dirs, files in os.walk(lits_input_path):
    for name in files:
        if name.endswith(".nii"):
            full_path=os.path.join(root, name)
            if "segmentation" in name:
                idx = name[13:-4]
                out_name = "slk_%03.0d.nii.gz" % int(idx)
                out_path = os.path.join(output_path, "labelsTr", out_name)
                src_path = os.path.join(root, name)
                img = nibabel.load(src_path)
                data = img.get_fdata()
                data[data >= 1] = 2 # LITS mask = 2
                new_img = nibabel.Nifti1Image(data.astype(numpy.float), img.affine)
                print("Copying " + src_path + " to " + out_path)
                nibabel.save(new_img, out_path)
                # shutil.copy(src_path, out_path)
            if "volume" in name:
                idx = name[7:-4]
                out_name = "slk_%03.0d.nii.gz" % int(idx)
                out_path = os.path.join(output_path, "imagesTr", out_name)
                src_path = os.path.join(root, name)
                print("Copying " + src_path + " to " + out_path)
                shutil.copy(src_path, out_path)

for root, dirs, files in os.walk(kits_input_path):
    for dir in dirs:
        if "case" in dir:
            idx = dir[-3:]
            for root2, dirs2, files2 in os.walk(os.path.join(kits_input_path, dir)):
                if len(files2) == 2:
                    for file2 in files2:
                        if "imaging" in file2:
                            out_name = "slk_%03.0d.nii.gz" % (int(idx) + 200)
                            out_path = os.path.join(output_path, "imagesTr", out_name)
                            src_path = os.path.join(root2, file2)
                            print("Copying " + src_path + " to " + out_path)
                            shutil.copy(src_path, out_path)
                        else:
                            out_name = "slk_%03.0d.nii.gz" % (int(idx) + 200)
                            out_path = os.path.join(output_path, "labelsTr", out_name)
                            src_path = os.path.join(root2, file2)
                            img = nibabel.load(src_path)
                            data = img.get_fdata()
                            data[data >= 1] = 3 # KITS mask = 3
                            new_img = nibabel.Nifti1Image(data.astype(numpy.float), img.affine)
                            print("Copying " + src_path + " to " + out_path)
                            nibabel.save(new_img, out_path)
                            #shutil.copy(src_path, out_path)

for root, dirs, files in os.walk(os.path.join(spleen_input_path, "imagesTr")):
    for file in files:
        if file.startswith("s"):
            idx = file[7:-7]
            out_name = "slk_%03.0d.nii.gz" % (int(idx) + 500)
            out_path = os.path.join(output_path, "imagesTr", out_name)
            src_path = os.path.join(root, file)
            print("Copying " + src_path + " to " + out_path)
            shutil.copy(src_path, out_path)
for root, dirs, files in os.walk(os.path.join(spleen_input_path, "labelsTr")):
    for file in files:
        if file.startswith("s"):
            idx = file[7:-7]
            out_name = "slk_%03.0d.nii.gz" % (int(idx) + 500)
            out_path = os.path.join(output_path, "labelsTr", out_name)
            src_path = os.path.join(root, file)
            print("Copying " + src_path + " to " + out_path)
            shutil.copy(src_path, out_path)