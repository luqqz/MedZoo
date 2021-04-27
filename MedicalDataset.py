import os
import numpy
import nibabel

from torch.utils import data

_image_target_size = (256, 256, 64)
_min_hu = 325
_max_hu = 325

class MedicalDataset(data.Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = []
        dirs, files = os.walk(self.root_dir)
        for root, dirs, files in os.walk(root_dir, topdown=False):
            for name in files:
                if 'segmentation' in name and (name.endswith('.nii') or name.endswith('.nii.gz')):
                    path = os.path.join(root, name)
                    self.files.append({
                        "image_path": path,
                        "label_path": path.replace('segmentation', 'volume')
                    })
                    print(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = self.files[index]
        ct_image = nibabel.load(item["image_path"])
        ct_label = nibabel.load(item["label_path"])
        ct_image = ct_image.get_fdata()
        ct_label = ct_label.get_fdata()

        ct_image = self.add_padding(ct_image, _image_target_size)
        ct_label = self.add_padding(ct_label, _image_target_size)



    def add_padding(self, ct_image, target_size):
        rows_to_add = target_size[0] - ct_image.shape[0]
        cols_to_add = target_size[1] - ct_image.shape[1]
        slices_to_add = target_size[2] - ct_image.shape[2]

        if rows_to_add < 0: rows_to_add = 0
        if cols_to_add < 0: cols_to_add = 0
        if slices_to_add < 0: slices_to_add = 0

        return numpy.pad(ct_image, ((0, rows_to_add), (0, cols_to_add), (0, slices_to_add)), 'constant')

    def normalize(self, ct_image):


a = MedicalDataset("E:\Studia\MGR\data\LITS")
print(a[1])
pass