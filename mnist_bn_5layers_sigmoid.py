import tensorflow as tf
import tensorflowvisu
import math
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)
# train/test selector for batch normalisation
tst = tf.placeholder(tf.bool)
# training iteration
iter = tf.placeholder(tf.int32)

# five layer and their number of neurons (last layer has 10 softmax neurons)
L = 200
M = 100
N = 60
P = 30
Q = 10

W1 = tf.Variable(tf.truncated_normal([784, L], stddev = 0.1))
S1 = tf.Variable(tf.ones([L]))
O1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
S2 = tf.Variable(tf.ones([M]))
O2 = tf.Variable(tf.zeros([M]))
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
S3 = tf.Variable(tf.ones([N]))
O3 = tf.Variable(tf.zeros([N]))
W4 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1))
S4 = tf.Variable(tf.ones([P]))
O4 = tf.Variable(tf.zeros([P]))
W5 = tf.Variable(tf.truncated_normal([P, Q], stddev=0.1))
B5 = tf.Variable(tf.zeros([Q]))

def batchnorm(Ylogits, Offset, Scale, is_test, iteration):
	exp_moving_avg = tf.train.ExponentialMovingAverage(0.998, iteration)
	bnepsilon = 1e-5
	mean, variance = tf.nn.moments(Ylogits, [0])
	update_moving_averages = exp_moving_avg.apply([mean, variance])
	m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
	v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
	Ybn = tf.nn.batch_normalization(Ylogits, m, v, Offset, Scale, bnepsilon)
	return Ybn, update_moving_averages

def no_batchnorm(Ylogits, Offset, Scale, is_test, iteration):
	return Ylogits, tf.no_op()


# model
XX = tf.reshape(X, [-1, 784])
Y1l = tf.matmul(XX, W1)
Y1bn, update_ema1 = batchnorm(Y1l, O1, S1, tst, iter)
Y1 = tf.nn.sigmoid(Y1bn)

Y2l = tf.matmul(Y1, W2)
Y2bn, update_ema2 = batchnorm(Y2l, O2, S2, tst, iter)
Y2 = tf.nn.sigmoid(Y2bn)

Y3l = tf.matmul(Y2, W3)
Y3bn, update_ema3 = batchnorm(Y3l, O3, S3, tst, iter)
Y3 = tf.nn.sigmoid(Y3bn)

Y4l = tf.matmul(Y3, W4)
Y4bn, update_ema4 = batchnorm(Y4l, O4, S4, tst, iter)
Y4 = tf.nn.sigmoid(Y4bn)

Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

# cross-entropy loss function (= -sum(Y_i, log(Yi)))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 and 1
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualization
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(O1, [-1]), tf.reshape(O2, [-1]), tf.reshape(O3, [-1]), tf.reshape(O4, [-1]), tf.reshape(B5, [-1])], 0)
# to use for sigmoid
allactivations = tf.concat([tf.reshape(Y1, [-1]), tf.reshape(Y2, [-1]), tf.reshape(Y3, [-1]), tf.reshape(Y4, [-1])], 0)
# to use for RELU
#allactivations = tf.concat([tf.reduce_max(Y1, [0]), tf.reduce_max(Y2, [0]), tf.reduce_max(Y3, [0]), tf.reduce_max(Y4, [0])], 0)
alllogits = tf.concat([tf.reshape(Y1l, [-1]), tf.reshape(Y2l, [-1]), tf.reshape(Y3l, [-1]), tf.reshape(Y4l, [-1])], 0)
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
datavis = tensorflowvisu.MnistDataVis(title4="Logits", title5="activations", histogram4colornum=2, histogram5colornum=2)

# training step learning rate = 0.003
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # learning rate decay (without batch norm)
    #max_learning_rate = 0.003
    #min_learning_rate = 0.0001
    #decay_speed = 2000
    # learning rate decay (with batch norm)
    max_learning_rate = 0.03
    min_learning_rate = 0.0001
    decay_speed = 1000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, al, ac = sess.run([accuracy, cross_entropy, I, alllogits, allactivations], {X: batch_X, Y_: batch_Y, tst: False})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        datavis.append_training_curves_data(i, a, c)
        datavis.update_image1(im)
        datavis.append_data_histograms(i, al, ac)

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], {X: mnist.test.images, Y_: mnist.test.labels, tst: True})
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        
    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False})
    sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i})


datavis.animate(training_step, iterations=2000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

