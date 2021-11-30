import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy

from models.tensorflow.unet.model import *
from datasets.tensorflow.MedicalDataset.MedicalDataset import MedicalDataset

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

NUM_THREADS = 4

medical_dataset = MedicalDataset(organ_list=['spleen'], crop_to_mask=False, data3d=False)
dataset = tf.data.Dataset.from_generator(medical_dataset.generate_data, output_types=(numpy.float32, numpy.float32))
dataset = dataset.map(lambda x,y : (x,y), num_parallel_calls=NUM_THREADS).prefetch(buffer_size=512)
dataset = dataset.batch(32)

model = unet(input_size=(192, 192, 1))

history = model.fit(
    medical_dataset.generate_data(),
    batch_size=1,
    epochs=10,
    steps_per_epoch=500,
    use_multiprocessing=True,
    workers=6
)
