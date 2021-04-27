import os
import numpy
import nibabel

from torch.utils import data

_image_target_size = (256, 256, 64)
_min_hu = -325
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
        ct_image = self.normalize(ct_image)

        ct_label = self.add_padding(ct_label, _image_target_size)

        ct_image = ct_image.transpose((0, 3, 1, 2)) # CDHW
        ct_label = ct_label.transpose((0, 3, 1, 2)) # DHW

        return ct_image.copy().astype(numpy.float32), ct_label.copy().astype(numpy.float32), 1

    def add_padding(self, ct_image, target_size):
        missing_rows = target_size[0] - ct_image.shape[0]
        missing_cols = target_size[1] - ct_image.shape[1]
        missing_slices = target_size[2] - ct_image.shape[2]

        if missing_rows < 0: missing_rows = 0
        if missing_cols < 0: missing_cols = 0
        if missing_slices < 0: missing_slices = 0

        return numpy.pad(ct_image, ((0, missing_rows), (0, missing_cols), (0, missing_slices)), 'constant')

    def normalize(self, ct_image):
        ct_image[numpy.where(ct_image < _min_hu)] = _min_hu
        ct_image[numpy.where(ct_image > _max_hu)] = _max_hu

        return (ct_image / float(_max_hu))


a = MedicalDataset("E:\Studia\MGR\data\LITS")
print(a[1])
pass