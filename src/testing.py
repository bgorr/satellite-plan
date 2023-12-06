import tensorflow as tf



# zeros in shape (32, 5)
tensor = tf.random.uniform([32, 5], maxval=1, dtype=tf.float32)
print(tensor)
batch_actions = tf.random.uniform([32], maxval=4, dtype=tf.int32)
# tensor = tf.gather(tensor, batch_actions, batch_dims=1)  # shape (32,)
tensor = tf.reduce_max(tensor, axis=1)
print(tensor)