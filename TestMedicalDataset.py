import os
from batchgenerators import transforms
import numpy
import math
import nibabel
import random
import cv2
import time

from batchgenerators.transforms import *

from utils.edatatype import DataType
from utils.eorgan import Organ

_final_data_processed_path = 'E:/Studia/MGR/MedZoo/data_final_processed/'
_image_target_size = [192, 192, 64] # H x W x D
_min_hu = -325
_max_hu = 325

_spleen_mask = 1
_kidney_mask = [2, 3]
_liver_mask = 6

class TestMedicalDataset:

    def __init__(self, data_root_dir=_final_data_processed_path, target_size=_image_target_size, crop_to_mask=False, transform=False):
        self.target_size = target_size
        self.crop_to_mask = crop_to_mask
        self.transform = transform
        self.files = []
        self.boundaries = {}

        for root, _, files in os.walk(data_root_dir):
            if Organ.BTCV.value in root:
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
                        elif Organ.BTCV.value in file:
                            task = Organ.BTCV

                        self.files.append({
                            'image_path': image_path,
                            'mask_path': mask_path,
                            'task': task
                        })

    def normalize(self, image):
        image[numpy.where(image < _min_hu)] = _min_hu
        image[numpy.where(image > _max_hu)] = _max_hu

        return (image / _max_hu)

    def add_padding(self, ct_image):
        print("ct image shape: " + str(ct_image.shape))
        missing_rows = 0
        missing_cols = 0
        if ct_image.shape[1] < ct_image.shape[0]:
            missing_rows = ct_image.shape[0] - ct_image.shape[1]
        else:
            missing_cols = ct_image.shape[1] - ct_image.shape[0]

        return numpy.pad(ct_image, ((0, missing_rows), (0, missing_cols), (0, 0)), 'constant')

    def get_organ_mask(self, mask, task):
        if task == Organ.KIDNEY or task == Organ.LIVER or task == Organ.SPLEEN:
            organ = (mask >= 1)
        elif task == Organ.BTCV:
            organ = (mask == _spleen_mask)
            organ += (mask == _kidney_mask[0])
            organ += (mask == _kidney_mask[1])
            organ += (mask == _liver_mask)
        else:
            print("Task not supported!")
            return None

        shape = mask.shape
        organ_mask = numpy.zeros((1, shape[0], shape[1], shape[2])).astype(numpy.float32)
        organ_mask[:, :, :] = numpy.where(organ, 1, -1)

        return organ_mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = self.files[index]
        image = nibabel.load(item['image_path'])
        mask = nibabel.load(item['mask_path'])
        task = item['task']

        image_data = image.get_fdata()
        mask_data = mask.get_fdata()

        print("image_data shape: " + str(image_data.shape))

        image_data = self.normalize(image_data)

        if Organ.KIDNEY.value not in item['image_path']:
            image_data = image_data.transpose((2, 1, 0))
            mask_data = mask_data.transpose((2, 1, 0))

        mask_data = self.get_organ_mask(mask_data, task)

        image_data = image_data[numpy.newaxis, :]

        image_data = image_data.astype(numpy.float32)
        mask_data = mask_data.astype(numpy.float32)

        return image_data.copy().astype(numpy.float32), mask_data.copy().astype(numpy.float32), task

    def get_generator(self):
        while True:
            shuffled_list = list(range(len(self.files)))
            random.shuffle(shuffled_list)

            for i in range(len(shuffled_list)):
                image, mask, task = self.__getitem__(shuffled_list[i])
                image = image[0, :, :, :]
                mask = mask[0, :, :, :]
                print(image.shape)
                print(mask.shape)

                # for j in range(image.shape[0] // 10):
                #     rand_h = random.randint(0, image.shape[1] - _image_target_size[0])
                #     rand_w = random.randint(0, image.shape[2] - _image_target_size[1])
                #     print("rand_h: %d" % rand_h)
                #     print("rand_w: %d" % rand_w)

                    # image_data = image[:, rand_h:rand_h + _image_target_size[0], rand_w:rand_w + _image_target_size[1]]
                    # mask_data = mask[:, rand_h:rand_h + _image_target_size[0], rand_w:rand_w + _image_target_size[1]]

                img_stack = numpy.zeros((image.shape[0], _image_target_size[0], _image_target_size[1]))
                mask_stack = numpy.zeros((mask.shape[0], _image_target_size[0], _image_target_size[1]))

                for k in range(image.shape[0]):
                    im = image[k, :, :]
                    ma = mask[k, :, :]
                    im = cv2.resize(im, (_image_target_size[0], _image_target_size[1]))
                    ma = cv2.resize(ma, (_image_target_size[0], _image_target_size[1]))
                    img_stack[k, :, :] = im
                    mask_stack[k, :, :] = ma

                img_stack = img_stack[:, :, :, numpy.newaxis]
                mask_stack = mask_stack[:, :, :, numpy.newaxis]

                yield (img_stack, mask_stack)

#
# use example
#

# a = MedicalDataset()
# import matplotlib.pyplot as plt

# for ex in a.get_generator():
#     plt.subplot(1, 2, 1)
#     plt.imshow(ex['data'][32, :, :])
#
#     plt.subplot(1,2,2)
#     plt.imshow(ex['mask'][32, :, :])
#     plt.show()