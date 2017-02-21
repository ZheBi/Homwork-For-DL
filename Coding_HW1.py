import tensorflow as tf
import numpy as np
import csv



x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


def write_csv(data):
    np.savetxt('output.csv', data, fmt ='%i', delimiter=',')

def weights_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2X2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def next_batch(data, i):
    batch= ([])
    input = data[50*(i%640):50*(i%640+1), 1:]
    label_maxtrix = np.zeros([50,10])
    t = 0
    for label in data[50*(i%640):50*(i%640+1),0]:
        label_maxtrix[t, int(label)]=1
        t += 1
    batch.append(input)
    batch.append(label_maxtrix)
    return batch

# First Layer of CNN
W_conv1 = weights_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv_2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2X2(h_conv1)

# Second Layer of CNN
W_conv2 = weights_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv_2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2X2(h_conv2)

# Full Connected Layer
W_fc1 = weights_variable([7*7*64, 1024])
b_fc1 = weights_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout Function
keep_pro = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_pro)

# Readout Layer
W_fc2 = weights_variable([1024, 10])
b_fc2 = weights_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2)+b_fc2

# Train the weights
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
output = tf.argmax(y_conv, 1)

# -------------------------------

if __name__ == "__main__":
    csv_train_path = "/Users/bjj/Documents/Big Data/train.csv"
    csv_test_path = "/Users/bjj/Documents/Big Data/test.csv"
    data_train = np.loadtxt(open(csv_train_path, "rb"), delimiter=",", skiprows=1)
    x_test = np.loadtxt(open(csv_test_path, "rb"), delimiter=",",skiprows=1)
    data_train.reshape([-1,785])
    x_test.reshape([-1,784])
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(10000):
            batch= next_batch(data_train, i)
            train_step.run(feed_dict={x: batch[0], y: batch[1], keep_pro: 0.5})
            print i
        Label = np.transpose(output.eval(feed_dict={x:x_test, keep_pro:0.5}))
        image_id = np.array(range(1, 10001))
        out_data = np.append(image_id, Label).reshape(-1, 10000).transpose()
        write_csv(out_data)
