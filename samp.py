import tensorflow as tf

# Check if Sequential, Dense, and Dropout exist
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    print("✅ Sequential, Dense, and Dropout are available in TensorFlow Keras!")
except ImportError as e:
    print("❌ Some required components are missing:", e)

# Print TensorFlow and Keras versions
print("TensorFlow Version:", tf.__version__)
print("Keras Version:", tf.keras.__version__)
