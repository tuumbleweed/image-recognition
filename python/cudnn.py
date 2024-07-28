import tensorflow as tf

# Print TensorFlow version
print("TensorFlow Version:", tf.__version__)

# Check if GPU is available
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu_devices))

# Print GPU details
for device in gpu_devices:
    print(device)
