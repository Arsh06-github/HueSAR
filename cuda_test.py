import torch
import tensorflow as tf
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

print(tf.config.list_physical_devices('GPU'))