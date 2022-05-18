from deepxde.backend import tf
import numpy as np

def tf_session():
    # tf session
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.Session(config=config)
    
    # init
    init = tf.global_variables_initializer()
    sess.run(init)
    
    return sess

def relative_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(tf.reduce_mean(tf.square(pred - exact))/tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))

def relative_error_(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt(np.sum(np.square(pred - exact))/np.sum(np.square(exact)))
    return tf.sqrt(tf.square(pred - exact)/tf.square(exact))

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.square(pred - exact))

def mean_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(pred - exact)
    return tf.reduce_mean(pred - exact)

def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x

def initialize_NN(self, layers):
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0, num_layers - 1):
        W = self.xavier_init(size=[layers[l], layers[l + 1]])
        b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)
    return weights, biases

def xavier_init(self, size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

"""def neural_net(self, X, weights, biases):
    num_layers = len(weights) + 1

    H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
    for l in range(0, num_layers - 2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y"""

def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        return False

class neural_net(object):
    def __init__(self, *inputs, layers):
        
        self.layers = layers
        self.num_layers = len(self.layers)
        #inputs = inputs.np()
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)
            self.X_min = X.min(0, keepdims=True)
            self.X_max = X.max(0, keepdims=True)
        
        self.weights = []
        self.biases = []
        self.gammas = []

        for l in range(0,self.num_layers-1):
            in_dim = self.layers[l]
            out_dim = self.layers[l+1]
            W = self.xavier_init(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def __call__(self, *inputs):
                
        H = 2 * (tf.concat(inputs, 1) - self.X_min)/(self.X_max - self.X_min) - 1 # data normalization, 方法不一样， 标准化与归一化
        #H = (tf.concat(inputs, 1) - self.X_mean) / self.X_std

        for l in range(0, self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W/tf.norm(W, axis = 0, keepdims=True)   # 这里多了weight normalization
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g*H + b
            # activation
            if l < self.num_layers-2:
                H = tf.tanh(H)
        if H.shape[1] == 3:
            Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
            return Y
        else:
            return H
