import tensorflow as tf

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check if it’s built with GPU support
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"GPU available: {tf.test.is_gpu_available()}")