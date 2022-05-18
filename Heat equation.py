from deepxde.backend import tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import time
from utilities import relative_error_, mkdir

class Heat_E:
    #Initialize the class
    def __init__(self, x, xb, layers, learning_rate):
        self.lb = xb.min(0)
        self.ub = xb.max(0)
        self.x = x
        self.xb = xb
        self.a = a
        self.L = 1.0
        self.n = 1.0
        self.layers = layers
        self.learning_rate = learning_rate
        # initialize the nn
        self.weights, self.biases = self.initialize_NN(layers)
        # tf placeholders
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.xb_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        # tf Graphs
        [self.y_pre, self.yt_pre, self.yxx_pre] = self.net_y(self.x_tf)
        [self.yb_pre, _, _] = self.net_y(self.xb_tf)

        # loss
        self.Heat_cond = self.Heat_conduction(self.yt_pre, self.yxx_pre)
        self.u_initial = self.initial_condition(self.xb)
        self.loss = tf.reduce_mean(tf.square(self.yb_pre[100:])) + \
                    tf.reduce_mean(tf.square((self.u_initial - self.yb_pre[:100]))) + \
                    tf.reduce_mean(tf.square(self.Heat_cond))

        # tensorboard
        self.L_eq = tf.reduce_mean(tf.square(self.Heat_cond))
        self.L_initial = tf.reduce_mean(tf.square(self.u_initial - self.yb_pre[:100]))
        self.L_boud = tf.reduce_mean(tf.square(self.yb_pre[100:]))
        tf.summary.scalar('loss_eq', self.L_eq)
        tf.summary.scalar('loss_initial', self.L_initial)
        tf.summary.scalar('loss_boundary', self.L_boud)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.summ = tf.summary.merge_all()
        self.tenboard_dir = './tensorboard/test/'
        self.writer = tf.summary.FileWriter(self.tenboard_dir, self.sess.graph)

        init = tf.global_variables_initializer()
        self.sess.run(init)

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

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_y(self, x):
        y = self.neural_net(x, self.weights, self.biases)
        y_t = dde.grad.jacobian(y, x, i=0, j=0)
        y_xx = dde.grad.hessian(y, x, i=1, j=1)

        return y, y_t, y_xx

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def callback(self, loss):
        print('loss:', loss)

    def train(self, nIter):

        tf_dict = {self.x_tf: self.x, self.xb_tf: self.xb}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                s = self.sess.run(self.summ, tf_dict)
                self.writer.add_summary(s, it)
                print('It: %d, Loss: %.4e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

    def initial_condition(self, X):
        x = X[:100, 1:2]
        y = np.sin(n * np.pi * x / L)

        return y

    def Heat_conduction(self, y_t, y_xx):
        return y_t - a*y_xx

    def heat_eq_exact_solution(self, x, t):

        return np.exp(-(n ** 2 * np.pi ** 2 * a * t) / (L ** 2)) * np.sin(n * np.pi * x / L)

    def gen_exact_solution(self):
        """
        Generates exact solution for the heat equation for the given values of x and t.
        """

        # Number of points in each dimension:
        x_dim, t_dim = (256, 201)

        # Bounds of 'x' and 't':
        x_min, t_min = (0, 0.)
        x_max, t_max = (L, 1.)

        # Create tensors:
        t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
        x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
        usol = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)

        # Obtain the value of the exact solution for each generated point:
        for i in range(x_dim):
            for j in range(t_dim):
                usol[i][j] = self.heat_eq_exact_solution(x[i], t[j])

        # Save solution:
        np.savez('heat_eq_data', x=x, t=t, usol=usol)
        data = np.load('heat_eq_data.npz')

    def gen_testdata(self):
        """
        Import and preprocess the dataset with the exact solution.
        """

        # Load the data:
        data = np.load('heat_eq_data.npz')

        # Obtain the values for t, x, and the excat solution:
        t, x, exact = data["t"], data["x"], data["usol"].T

        # Process the data and flatten it out (like labels and features):
        xx, tt = np.meshgrid(x, t)
        X = np.vstack((np.ravel(tt), np.ravel(xx))).T
        y = exact.flatten()[:, None]

        return X, y, xx, tt

    def predict(self, x_star):
        tf_dict = {self.x_tf: x_star}
        y_star = self.sess.run(self.y_pre, tf_dict)
        return y_star

if __name__ == "__main__":
    a = 0.4  # Thermal diffusivity
    L = 1.  # Length of the bar
    n = 1  # Frequency of the sinusoidal initial conditions
    lb = np.array([[0.], [0.]])
    ub = np.array([[L], [1.]])
    layers = [2] + [20] * 3 + [1]
    # domain data
    x_domain = np.random.random((2000, 1))
    t_domain = np.random.random((2000, 1))
    X = np.hstack((t_domain, x_domain))
    # boudary data
    t_1 = np.full((100, 1), 0.)
    t_2 = np.random.random((50, 1))
    t_3 = np.random.random((50, 1))
    t_b = np.vstack((t_1, t_2, t_3))
    x_1 = np.random.random((100, 1))
    x_2 = np.full((50, 1), 0.)
    x_3 = np.full((50, 1), L)
    x_b = np.vstack((x_1, x_2, x_3))
    X_b = np.hstack((t_b, x_b))

    model = Heat_E(X, X_b, layers=layers, learning_rate=0.001)
    model.train(1)

    model.gen_exact_solution()
    [X_t, u, xx, tt] = model.gen_testdata()
    u_test = model.predict(X_t)
    u_exact = model.heat_eq_exact_solution(X_t[:, 1], X_t[:, 0])

    error = relative_error_(u_test, u_exact)
    print('Error u: %e' % (error))

    SavePath = './Result/'
    mkdir(SavePath)

    """# plot
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    fig = plt.figure()
    ax1 = plt.subplot(121)
    U_test = u_test.reshape((201, 256))
    levels = np.linspace(u_test.min(), u_test.max(), 30)
    gra = plt.contourf(xx, tt, U_test, cmap='jet', levels=levels)
    fig.colorbar(gra, ax=ax1)
    ax1.set(title='PINN', xlabel='x', ylabel='t')

    ax2 = plt.subplot(122)
    U_exact = u_exact.reshape((201, 256))
    levels = np.linspace(u_exact.min(), u_exact.max(), 30)
    gra = plt.contourf(xx, tt, U_exact, cmap='jet', levels=levels)
    fig.colorbar(gra, ax=ax2)
    ax2.set(title='Exact', xlabel='x', ylabel='t')

    ax3 = plt.subplot(133)
    u_dif = np.abs(u_test - u_exact)
    U_dif = u_dif.reshape((201, 256))
    levels = np.linspace(u_dif.min(), u_dif.max(), 30)
    gra = plt.contourf(xx, tt, U_dif, cmap='jet', levels=levels)
    fig.colorbar(gra, ax=ax3)
    ax3.set(title='Error', xlabel='x', ylabel='t')
    plt.subplots_adjust(wspace=0.5)
    plt.savefig(SavePath + 'Heat Equation_Exact Solution.png', dpi=150)

    plt.show()"""
