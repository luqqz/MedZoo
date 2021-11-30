import tensorflow_datasets as tfds
import os
import numpy
import math
import nibabel
import random
import cv2

from batchgenerators.transforms import *

from enums.datatype import DataType
from enums.organ import Organ

_final_data_processed_path = '/notebooks/MedZoo-main/data_final_processed/'
_image_target_size = [192, 192, 64]  # H x W x D
_min_hu = -325
_max_hu = 325


class MedicalDataset:

    def __init__(self, data_root_dir=_final_data_processed_path, target_size=_image_target_size, organ_list=["spleen", "kidney", "liver"], crop_to_mask=True, crop_depth_to_mask=True, data3d=False, transform=False):
        self.target_size = target_size
        self.organ_list = organ_list
        self.crop_to_mask = crop_to_mask
        self.crop_depth_to_mask = crop_depth_to_mask
        self.data3d = data3d
        self.transform = transform
        self.slices = 0
        self.files = []
        self.boundaries = {}

        for root, _, files in os.walk(data_root_dir):
            if 'train' in root:
                for file in files:
                    if 'ct' in file:
                        image_path = os.path.join(root, file)
                        mask_path = image_path.replace('ct', 'mask')

                        if Organ.KIDNEY.value in file:
                            if not Organ.KIDNEY.value in self.organ_list:
                                continue
                            task = Organ.KIDNEY
                        elif Organ.LIVER.value in file:
                            if not Organ.LIVER.value in self.organ_list:
                                continue
                            task = Organ.LIVER
                        elif Organ.SPLEEN.value in file:
                            if not Organ.SPLEEN.value in self.organ_list:
                                continue
                            task = Organ.SPLEEN

                        if not self.crop_to_mask:
                            img = nibabel.load(image_path)
                            self.slices += img.header.get_data_shape()[2]

                        self.files.append({
                            'image_path': image_path,
                            'mask_path': mask_path,
                            'task': task
                        })

    def normalize(self, image):
        image[numpy.where(image < _min_hu)] = _min_hu
        image[numpy.where(image > _max_hu)] = _max_hu

        return (image / _max_hu)

    def add_padding(self, ct_image, target_size):
        missing_rows = target_size[0] - ct_image.shape[0]
        missing_cols = target_size[1] - ct_image.shape[1]
        missing_slices = target_size[2] - ct_image.shape[2]

        if missing_rows < 0:
            missing_rows = 0
        if missing_cols < 0:
            missing_cols = 0
        if missing_slices < 0:
            missing_slices = 0

        return numpy.pad(ct_image, ((0, missing_rows), (0, missing_cols), (0, missing_slices)), 'constant')

    def locate_boundaries(self, mask, index, random_crop_prob=0.2):
        img_w, img_h, img_d = mask.shape
        margin = 16

        if index in self.boundaries.keys():
            boundaries = self.boundaries[index]

            boundary_w_min = boundaries[0]
            boundary_w_max = boundaries[1]
            boundary_h_min = boundaries[2]
            boundary_h_max = boundaries[3]
            boundary_d_min = boundaries[4]
            boundary_d_max = boundaries[5]
        if not index in self.boundaries.keys():
            boundary_w, boundary_h, boundary_d = numpy.where(mask >= 1)

            boundary_w_min = boundary_w.min()
            boundary_w_max = boundary_w.max()
            boundary_h_min = boundary_h.min()
            boundary_h_max = boundary_h.max()
            boundary_d_min = boundary_d.min()
            boundary_d_max = boundary_d.max()

            self.boundaries[index] = (
                boundary_w_min,
                boundary_w_max,
                boundary_h_min,
                boundary_h_max,
                boundary_d_min,
                boundary_d_max
            )

        if (boundary_w_max - boundary_w_min) < self.target_size[0]:
            missing = self.target_size[0] - (boundary_w_max - boundary_w_min)
            missing_min = missing // 2
            if missing % 2 == 0:
                missing_max = (missing // 2) + 1
            else:
                missing_max = missing_min

            boundary_w_max = boundary_w_max + missing_max
            boundary_w_min = boundary_w_min - missing_min
            if boundary_w_min < 0:
                boundary_w_min = 0
                boundary_w_max = numpy.min(self.target_size[0], img_w)

        if (boundary_h_max - boundary_h_min) < self.target_size[1]:
            missing = self.target_size[1] - (boundary_h_max - boundary_h_min)
            missing_min = missing // 2
            if missing % 2 == 0:
                missing_max = (missing // 2) + 1
            else:
                missing_max = missing_min

            boundary_h_max = boundary_h_max + missing_max
            boundary_h_min = boundary_h_min - missing_min
            if boundary_h_min < 0:
                boundary_h_min = 0
                boundary_h_max = numpy.min(self.target_size[1], img_h)

        if (boundary_d_max - boundary_d_min) < self.target_size[2]:
            missing = self.target_size[2] - (boundary_d_max - boundary_d_min)
            missing_min = missing // 2
            if missing % 2 == 0:
                missing_max = (missing // 2) + 1
            else:
                missing_max = missing_min

            boundary_d_max = boundary_d_max + missing_max
            boundary_d_min = boundary_d_min - missing_min
            if boundary_d_min < 0:
                boundary_d_min = 0
                boundary_d_max = numpy.min(self.target_size[2], img_d)

        boundary_w_target_min = numpy.max([boundary_w_min - margin, 0])
        boundary_w_target_max = numpy.min([boundary_w_max + margin, img_w])
        boundary_h_target_min = numpy.max([boundary_h_min - margin, 0])
        boundary_h_target_max = numpy.min([boundary_h_max + margin, img_h])
        boundary_d_target_min = numpy.max([boundary_d_min - margin, 0])
        boundary_d_target_max = numpy.min([boundary_d_max + margin, img_d])

        if random.random() < (1.0 - random_crop_prob):
            w0 = random.randint(
                boundary_w_target_min,
                numpy.max([boundary_w_target_max -
                          self.target_size[0], boundary_w_target_min])
            )
            h0 = random.randint(
                boundary_h_target_min,
                numpy.max([boundary_h_target_max -
                          self.target_size[1], boundary_h_target_min])
            )
            d0 = random.randint(
                boundary_d_target_min,
                numpy.max([boundary_d_target_max -
                          self.target_size[2], boundary_d_target_min])
            )
        else:
            w0 = random.randint(0, img_w - self.target_size[0])
            h0 = random.randint(0, img_h - self.target_size[1])
            d0 = random.randint(0, img_d - self.target_size[2])

        w1 = w0 + self.target_size[0]
        h1 = h0 + self.target_size[1]
        d1 = d0 + self.target_size[2]

        return [(w0, w1), (h0, h1), (d0, d1)]

    def get_organ_mask(self, mask, task):
        if task == Organ.KIDNEY or task == Organ.LIVER or task == Organ.SPLEEN:
            organ = (mask >= 1)
        else:
            print("Task not supported!")
            return None

        shape = mask.shape
        organ_mask = numpy.zeros(
            (shape[0], shape[1], shape[2])).astype(numpy.float32)
        organ_mask = numpy.where(organ, 1, 0)

        return organ_mask

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
            ),
        ])

    def get_steps_per_epoch(self, batch_size):
        if self.crop_to_mask and not self.data3d:
            return (len(self.files) * self.target_size[2]) // batch_size
        elif not self.data3d:
            return self.slices // batch_size
        return self.__len__()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = self.files[index]
        image = nibabel.load(item['image_path'])
        mask = nibabel.load(item['mask_path'])
        task = item['task']

        image_data = image.get_fdata()
        mask_data = mask.get_fdata()

        image_data = self.add_padding(image_data, self.target_size)
        image_data = self.normalize(image_data)

        mask_data = self.add_padding(mask_data, self.target_size)

        if self.crop_to_mask or self.crop_depth_to_mask:
            width, height, depth = self.locate_boundaries(mask_data, index)

            if self.crop_to_mask:
                image_data = image_data[width[0]: width[1],
                                        height[0]: height[1], depth[0]:depth[1]]
                mask_data = mask_data[width[0]: width[1],
                                        height[0]: height[1], depth[0]:depth[1]]
            else:
                image_data = image_data[:, :, depth[0]:depth[1]]
                mask_data = mask_data[:, :, depth[0]:depth[1]]
        if not self.crop_to_mask:
            image_data = cv2.resize(image_data, (self.target_size[0], self.target_size[1]))
            mask_data = cv2.resize(mask_data, (self.target_size[0], self.target_size[1]))

        image_data = image_data.transpose((2, 1, 0))
        mask_data = mask_data.transpose((2, 1, 0))

        mask_data = self.get_organ_mask(mask_data, task)

        #image_data = image_data[numpy.newaxis, :]

        image_data = image_data.astype(numpy.float32)
        mask_data = mask_data.astype(numpy.float32)
        print('image shape: ' + str(image_data.shape))
        print('mask shape: ' + str(mask_data.shape))

        return image_data.copy().astype(numpy.float32), mask_data.copy().astype(numpy.float32), task

    def generate_data(self):
        while True:
            shuffled_list = list(range(len(self.files)))
            random.shuffle(shuffled_list)
            print('Generate data, epoch start')

            for i in range(len(shuffled_list)):
                print('Loading example [' + str(shuffled_list[i]) + ']')
                image, mask, task = self.__getitem__(shuffled_list[i])
                image = image[:, :, :, numpy.newaxis]
                mask = mask[:, :, :, numpy.newaxis]
                data = {
                        'data': image,
                        'mask': mask,
                        'task': task
                }
                if self.transform:
                    transforms = self.get_transforms()
                    data = transforms(**data)

                if self.data3d:  # return whole 3D image at once
                    yield (data['data'], data['mask'])
                else: # yield slice by slice, D x W x H
                    for j in range(image.shape[0]):
                        yield (data['data'][j, :, :, :], data['mask'][j, :, :, :])
