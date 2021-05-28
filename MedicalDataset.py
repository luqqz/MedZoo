import os
import numpy
import nibabel

from batchgenerators.transforms import *

from utils.edatatype import DataType
from utils.eorgan import Organ

_final_data_processed_path = 'data_final_processed/'
_image_target_size = [128, 128, 64] # H x W x D
_min_hu = -325
_max_hu = 325

class MedicalDataset:

    def __init__(self, data_root_dir=_final_data_processed_path, crop_to_mask=False, transform=False):
        self.crop_to_mask = crop_to_mask
        self.transform = transform
        self.files = []

        for root, _, files in os.walk(data_root_dir):
            for file in files:
                if 'ct' in file:
                    image_path = os.path.join(root, file)
                    mask_path = image_path.replace('ct', 'mask')
                    if Organ.KIDNEY.value in file:
                        task = Organ.KIDNEY
                    elif Organ.LIVER.value in file:
                        task = Organ.LIVER
                    elif Organ.SPLEEN.value in file:
                        task = Organ.SPLEEN
                        
                    self.files.append({
                        'image_path': image_path,
                        'mask_path': mask_path,
                        'task': task
                    })

    def normalize(self, ct_image):
        ct_image[numpy.where(ct_image < _min_hu)] = _min_hu
        ct_image[numpy.where(ct_image > _max_hu)] = _max_hu

        return (ct_image / float(_max_hu))

    def add_padding(self, ct_image, target_size):
        missing_rows = target_size[0] - ct_image.shape[0]
        missing_cols = target_size[1] - ct_image.shape[1]
        missing_slices = target_size[2] - ct_image.shape[2]

        if missing_rows < 0: missing_rows = 0
        if missing_cols < 0: missing_cols = 0
        if missing_slices < 0: missing_slices = 0

        return numpy.pad(ct_image, ((0, missing_rows), (0, missing_cols), (0, missing_slices)), 'constant')

    def get_transforms(self):
        return Compose([
            GaussianNoiseTransform(
                p_per_sample=0.1
                ),
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.75, 1.25),
                p_per_sample=0.15
                ),
            BrightnessTransform(
                mu=0.0,
                sigma=0.1,
                per_channel=True,
                p_per_sample=0.15,
                p_per_channel=0.5
                ),
            ContrastAugmentationTransform(
                p_per_sample=0.15
                ),
            SimulateLowResolutionTransform(
                zoom_range=(.5, 1),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.25,
                ignore_axes=None
                ),
            GammaTransform(
                gamma_range=(.7, 1.5),
                invert_image=False,
                per_channel=True,
                retain_stats=True,
                p_per_sample=0.15
                )
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = self.files[index]
        image = nibabel.load(item['image_path'])
        mask = nibabel.load(item['mask_path'])
        task = item['task']

        image_data = image.get_fdata()
        mask_data = mask.get_fdata()

        image_data = self.add_padding(image_data, _image_target_size)
        image_data = self.normalize(image_data)

        mask_data = self.add_padding(mask_data, _image_target_size)

        image_data = image_data[numpy.newaxis, :]
        mask_data = mask_data[numpy.newaxis, :]

        if Organ.KIDNEY.value in item['image_path']:
            pass # if kidney - continue
        else:    # else - transpose
            image_data = image_data.transpose((0, 3, 2, 1))
            mask_data = mask_data.transpose((0, 3, 2, 1))

        return image_data.copy().astype(numpy.float32), mask_data.copy().astype(numpy.float32), task

a = MedicalDataset()
img = a[0][0]
import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.imshow(img[0,200,:, :])
data_dict = {'data': img}
tr = a.get_transforms()
b = tr(**data_dict)
plt.subplot(1,2,2)
plt.imshow(b['data'][0,200,:, :])
plt.show()