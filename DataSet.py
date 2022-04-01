import numpy as np
import tensorflow as tf

class DataSet:
    def __init__(self, x_range, y_range, Nx_train, Ny_train, N_bc):
        self.x_range = x_range
        self.y_range = y_range
        self.Nx_train = Nx_train
        self.Ny_train = Ny_train
        self.N_bc = N_bc
        self.e = np.array([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
        self.w = np.full((9, 1), 0.0)
        self.w[0] = 4 / 9
        self.w[1:5] = 1 / 9
        self.w[5:] = 1 / 36
        self.RT = 100
        self.xi = self.e * np.sqrt(3 * self.RT)
        self.Re = 20
        self.u_0 = 1.0
        self.p_0 = 1.0
        self.L = 1.
        self.nu = 1 / self.Re
        self.tau = 5e-4
        self.lamda = self.Re / 2 - np.sqrt(self.Re ** 2 / 4 + 4 * np.pi ** 2)
        self.sess = tf.Session()
        self.x_l = self.x_range.min()
        self.x_u = self.x_range.max()
        self.y_l = self.y_range.min()
        self.y_u = self.y_range.max()
    # xi_x * feq_x + xi_y * feq_y
    def feq_gradient(self, rou, u, v, x, y):

        rou_x = tf.gradients(rou, x)[0]
        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]

        rou_y = tf.gradients(rou, y)[0]
        u_y = tf.gradients(u, y)[0]
        v_y = tf.gradients(v, y)[0]
        f_sum = self.feq_xy(rou, u, v, rou_x, rou_y, u_x, v_x, u_y, v_y)
        return f_sum

    def feq_xy(self, rou, u, v, rou_x, rou_y, u_x, v_x, u_y, v_y):
        f_sum = self.dfeq_xy(rou, u, v, rou_x, rou_y, u_x, v_x, u_y, v_y, 0)
        for i in range(1, 9):
            f_ = self.dfeq_xy(rou, u, v, rou_x, rou_y, u_x, v_x, u_y, v_y, i)
            #f_sum = tf.concat([f_sum, f_], 1)
            f_sum = np.hstack((f_sum, f_))
        return f_sum

    def dfeq_xy(self, rou, u, v, rou_x, rou_y, u_x, v_x, u_y, v_y, i):
        feq_x = self.w[i, :] * rou_x * (1 + (self.xi[i, 0] * u + self.xi[i, 1] * v) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) ** 2 / 2 / self.RT ** 2 - (u ** 2 + v ** 2) / 2 / self.RT) + \
                self.w[i, :] * rou * ((self.xi[i, 0] * u_x + self.xi[i, 1] * v_x) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) * (self.xi[i, 0] * u_x + self.xi[i, 1] * v_x) / self.RT ** 2 - (u * u_x + v * v_x) / self.RT)

        feq_y = self.w[i, :] * rou_y * (1 + (self.xi[i, 0] * u + self.xi[i, 1] * v) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) ** 2 / 2 / self.RT ** 2 - (u ** 2 + v ** 2) / 2 / self.RT) + \
                self.w[i, :] * rou * ((self.xi[i, 0] * u_y + self.xi[i, 1] * v_y) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) * (self.xi[i, 0] * u_y + self.xi[i, 1] * v_y) / self.RT ** 2 - (u * u_y + v * v_y) / self.RT)
        dfeq_xy = self.xi[i, 0:1] * feq_x + self.xi[i, 1:2] * feq_y
        return dfeq_xy

    def d2feq_xy2(self, rou, u, v, x, y, i):
        feq_2 = self.xi[i, 0] ** 2 * (0.22945878296434 * rou * self.w[i] * (-1 / self.RT + self.xi[i, 0] ** 2 / self.RT ** 2) * np.exp(-6.05969085684496 * x) * np.cos(
            6.28318530717959 * y) ** 2 + 0.0533563232499575 * rou * self.w[i] * (-1 / self.RT + self.xi[i, 1] ** 2 / self.RT ** 2) * np.exp(
            -6.05969085684496 * x) * np.sin(6.28318530717959 * y) ** 2 - 1.45135220091297 * rou * self.w[i] * (
                                -u / self.RT + self.xi[i, 0] / self.RT + self.xi[i, 0] * (u * self.xi[i, 0] + v * self.xi[i, 1]) / self.RT ** 2) * np.exp(
            -3.02984542842248 * x) * np.cos(6.28318530717959 * y) - 0.699863622666408 * rou * self.w[i] * (
                                -v / self.RT + self.xi[i, 1] / self.RT + self.xi[i, 1] * (u * self.xi[i, 0] + v * self.xi[i, 1]) / self.RT ** 2) * np.exp(
            -3.02984542842248 * x) * np.sin(6.28318530717959 * y) - 1.83599266402652 * self.w[i] * (
                                1 - (u ** 2 / 2 + v ** 2 / 2) / self.RT + (u * self.xi[i, 0] + v * self.xi[i, 1]) / self.RT + (
                                    u * self.xi[i, 0] + v * self.xi[i, 1]) ** 2 / (2 * self.RT ** 2)) * np.exp(
            -6.05969085684496 * x) / self.RT) + 2 * self.xi[i, 0] * self.xi[i, 1] * (
                    0.47584343418981 * rou * self.w[i] * (-1 / self.RT + self.xi[i, 0] ** 2 / self.RT ** 2) * np.exp(-6.05969085684496 * x) * np.sin(
                6.28318530717959 * y) * np.cos(6.28318530717959 * y) - 0.110648438743559 * rou * self.w[i] * (
                                -1 / self.RT + self.xi[i, 1] ** 2 / self.RT ** 2) * np.exp(-6.05969085684496 * x) * np.sin(
                6.28318530717959 * y) * np.cos(6.28318530717959 * y) - 3.00976239209241 * rou * self.w[i] * (
                                -u / self.RT + self.xi[i, 0] / self.RT + self.xi[i, 0] * (u * self.xi[i, 0] + v * self.xi[i, 1]) / self.RT ** 2) * np.exp(
                -3.02984542842248 * x) * np.sin(6.28318530717959 * y) + 1.45135220091297 * rou * self.w[i] * (
                                -v / self.RT + self.xi[i, 1] / self.RT + self.xi[i, 1] * (u * self.xi[i, 0] + v * self.xi[i, 1]) / self.RT ** 2) * np.exp(
                -3.02984542842248 * x) * np.cos(6.28318530717959 * y)) + self.xi[i, 1] ** 2 * (
                    0.986787129855652 * rou * self.w[i] * (-1 / self.RT + self.xi[i, 0] ** 2 / self.RT ** 2) * np.exp(-6.05969085684496 * x) * np.sin(
                6.28318530717959 * y) ** 2 + 0.22945878296434 * rou * self.w[i] * (-1 / self.RT + self.xi[i, 1] ** 2 / self.RT ** 2) * np.exp(
                -6.05969085684496 * x) * np.cos(6.28318530717959 * y) ** 2 + 6.24153782324891 * rou * self.w[i] * (
                                -u / self.RT + self.xi[i, 0] / self.RT + self.xi[i, 0] * (u * self.xi[i, 0] + v * self.xi[i, 1]) / self.RT ** 2) * np.exp(
                -3.02984542842248 * x) * np.cos(6.28318530717959 * y) + 3.00976239209241 * rou * self.w[i] * (
                                -v / self.RT + self.xi[i, 1] / self.RT + self.xi[i, 1] * (u * self.xi[i, 0] + v * self.xi[i, 1]) / self.RT ** 2) * np.exp(
                -3.02984542842248 * x) * np.sin(6.28318530717959 * y))

        return feq_2

    def d2phi_xy2(self, x, y):
        rho_xx = -4*np.exp(2*self.lamda*x)*self.lamda**2*self.p_0/self.RT
        rho_yy = np.zeros_like(x)
        u_xx = -np.exp(self.lamda*x)*self.lamda**2*self.u_0*np.cos(2*np.pi*y)
        u_yy = 4*np.exp(self.lamda*x)*np.pi**2*self.u_0*np.cos(2*np.pi*y)
        v_xx = np.exp(self.lamda*x)*self.lamda**3*self.u_0*np.sin(2*np.pi*y)/2/np.pi
        v_yy = -2*np.exp(self.lamda*x)*self.lamda*np.pi*self.u_0*np.sin(2*np.pi*y)
        rho_xy = np.zeros_like(x)
        u_xy = 2*np.exp(self.lamda*x)*self.lamda*np.pi*self.u_0*np.sin(2*np.pi*y)
        v_xy = np.exp(self.lamda*x)*self.lamda**2*self.u_0*np.cos(2*np.pi*y)
        d2phi = np.hstack((rho_xx, rho_yy, u_xx, u_yy, v_xx, v_yy, rho_xy, u_xy, v_xy))
        return d2phi

    def dphi_xy(self, x, y):
        rou_x = (-2*self.lamda * self.p_0 * np.exp(2 * self.lamda * x)) / self.RT
        rou_y = np.zeros_like(x)
        u_x = -self.u_0 * self.lamda * np.cos(2 * np.pi * y) * np.exp(self.lamda * x)
        u_y = 2 * np.pi * self.u_0 * np.sin(2 * np.pi * y) * np.exp(self.lamda * x)
        v_x = self.lamda ** 2 / 2 / np.pi * self.u_0 * np.sin(2 * np.pi * y) * np.exp(self.lamda * x)
        v_y = self.lamda * self.u_0 * np.cos(2 * np.pi * y) * np.exp(self.lamda * x)

        dphi = np.hstack((rou_x, rou_y, u_x, u_y, v_x, v_y))
        return dphi

    def exa(self, x, y):
        p = self.p_func(x)
        rho = self.rou_func(p)
        u = self.u_train(x, y)
        v = self.v_train(x, y)
        dphi = self.dphi_xy(x, y)
        rho_x = dphi[:, 0:1]
        rho_y = dphi[:, 1:2]
        u_x = dphi[:, 2:3]
        u_y = dphi[:, 3:4]
        v_x = dphi[:, 4:5]
        v_y = dphi[:, -1]
        d2phi = self.d2phi_xy2(x, y)
        rho_xx = d2phi[:, 0:1]
        rho_yy = d2phi[:, 1:2]
        u_xx = d2phi[:, 2:3]
        u_yy = d2phi[:, 3:4]
        v_xx = d2phi[:, 4:5]
        v_yy = d2phi[:, 5:6]
        rho_xy = d2phi[:, 6:7]
        u_xy = d2phi[:, 7:8]
        v_xy = d2phi[:, -1]
        dfneq_sum = np.zeros_like(x)
        dfeq = self.feq_xy(rho, u, v, rho_x, rho_y, u_x, v_x, u_y, v_y)
        fneq = -self.tau*dfeq

        for i in range(9):
            d2feq = self.d2feq_xy2(rho, u, v, rho_x, rho_y, u_x, u_y, v_x, v_y, rho_xx, rho_yy, u_xx, u_yy, v_xx, v_yy, rho_xy, u_xy, v_xy, i)
            f_xx = d2feq[:, 0:1]
            f_yy = d2feq[:, 1:2]
            f_xy = d2feq[:, -1][:, None]
            dfneq = -self.tau*(self.xi[i, 0]**2*f_xx+2*self.xi[i, 0]*self.xi[i, 1]*f_xy+self.xi[i, 1]**2*f_yy)
            dfneq_sum = np.hstack((dfneq_sum, dfneq))
        dfneq = dfneq_sum[:, 1:]
        bgk_sum = np.zeros_like(x)
        for i in range(9):
            bgk = dfneq[:, i] + dfeq[:, i] + fneq[:, i] / self.tau
            # bgk_sum = tf.concat([bgk_sum, bgk[:, None]], 1)
            bgk_sum = np.hstack((bgk_sum, bgk[:, None]))
        return dfneq

    # feq_i
    def f_eq(self, rou, u, v):
        f_eq_sum = self.f_eqk(rou, u, v, 0)
        for i in range(1, 9):
            f_eq = self.f_eqk(rou, u, v, i)
            f_eq_sum = tf.concat([f_eq_sum, f_eq], 1)
        return f_eq_sum

    # f_eq equation
    def f_eqk(self, rou, u, v, k):
        f_eqk = self.w[k, :] * rou * (1 + (self.xi[k, 0]*u + self.xi[k, 1]*v) / self.RT + (self.xi[k, 0]*u + self.xi[k, 1]*v) ** 2 / 2 / self.RT ** 2 - (u*u + v*v) / 2 / self.RT)
        return f_eqk

    # BGK equation
    def bgk(self, f_neq, rou, u, v, x, y):
        feq_pre = self.feq_gradient(rou, u, v, x, y)
        R_sum = 0
        for k in range(9):
            fneq_x = tf.gradients(f_neq[:, k][:, None], x)[0]
            fneq_y = tf.gradients(f_neq, y)[0]
            R = (self.xi[k, 0] * fneq_x + self.xi[k, 1] * fneq_y + feq_pre[:, k][:, None] + 1 / self.tau * (f_neq[:, k][:, None])) ** 2
            R_sum = R_sum + R
        return R_sum

    # the equation residual
    def Eq_res(self, f_neq, rou, u, v, x, y):
        feq_pre = self.feq_gradient(rou, u, v, x, y)
        Eq_sum = x * 0
        for k in range(9):
            fneq_x = tf.gradients(f_neq, x)[0]
            fneq_y = tf.gradients(f_neq, y)[0]
            Eq = tf.abs(self.xi[k, 0] * fneq_x + self.xi[k, 1] * fneq_y + feq_pre[:, k][:, None] + 1 / self.tau * (f_neq[:, k][:, None]))
            Eq_sum = tf.concat([Eq_sum, Eq], 1)
        return Eq_sum

    # boundary condition
    # 判断速度与边界向内向量乘积
    def inward_judge(self, x, y):
        x = tf.where(tf.equal(x, 2.0), x * 0 - 3.0, x)
        x = tf.where(tf.equal(x, -0.5), x * 0 + 3.0, x)
        x = tf.where(tf.equal(tf.abs(x), 3.0), x / 3.0, x * 0.0)
        y = tf.where(tf.equal(y, 1.5), y * 0 - 3.0, y)
        y = tf.where(tf.equal(y, -0.5), y * 0 + 3.0, y)
        y = tf.where(tf.equal(tf.abs(y), 3.0), y / 3.0, y * 0.0)
        return x, y

    def bgk_cond(self, f_neq, feq_pre, x, y):
        [xx, yy] = self.inward_judge(x, y)
        R_sum = 0
        for k in range(9):
            fneq_x = tf.gradients(f_neq, x)[0]
            fneq_y = tf.gradients(f_neq, y)[0]
            cond = self.xi[k, 0] * xx + self.xi[k, 1] * yy
            cond = tf.squeeze(cond)
            R_ = (self.xi[k, 0] * fneq_x + self.xi[k, 1] * fneq_y + feq_pre[:, k][:, None] + 1 / self.tau * (
            f_neq[:, k][:, None])) ** 2
            R = tf.where(tf.greater(cond, cond * 0), R_ * 0, R_)
            R_sum = R_sum + R

        return R_sum

    # boundary condition for fneq
    def fBC(self, f_neq_nn, rou, u, v, x_bc, y_bc):
        feq_ex = self.Ex_fneq_(rou, u, v, x_bc, y_bc) # 计算xi_x * feq_x + xi_y * feq_y精确解
        fbc_sum = 0
        for i in range(9):
            f = (f_neq_nn[:, i][:, None] + self.tau * feq_ex[:, i][:, None] * 1e4) ** 2
            fbc_sum = fbc_sum + f
        return fbc_sum

    # exact solution
    def u_train(self, x, y):
        u = self.u_0 * (1 - np.exp(self.lamda * x) * np.cos(2 * np.pi * y))
        return u
    def v_train(self, x, y):
        v = self.u_0 * (self.lamda / (2 * np.pi) * np.exp(self.lamda * x) * np.sin(2 * np.pi * y))
        return v

    def p_func(self, x):
        p = self.p_0*(1 - 0.5 * np.exp(2 * self.lamda * x)) + self.RT
        return p

    def rou_func(self, p):
        rou = p / self.RT
        return rou

    # fneq求解公式中xi_x * feq_x + xi_y * feq_y部分计算
    def Ex_fneq_(self, rou, u, v, x, y):
        rou_x = (-self.lamda * self.p_0 * np.exp(2 * self.lamda * x)) / self.RT
        rou_y = 0.
        u_x = -self.u_0 * self.lamda * np.cos(2 * np.pi * y) * np.exp(self.lamda * x)
        u_y = 2 * np.pi * self.u_0 * np.sin(2 * np.pi * y) * np.exp(self.lamda * x)
        v_x = self.lamda ** 2 / 2 / np.pi * self.u_0 * np.sin(2 * np.pi * y) * np.exp(self.lamda * x)
        v_y = self.lamda * self.u_0 * np.cos(2 * np.pi * y) * np.exp(self.lamda * x)
        f_sum = self.feq_xy(rou, u, v, rou_x, rou_y, u_x, v_x, u_y, v_y)
        f_sum = tf.cast(f_sum, dtype=tf.float32)
        return f_sum
    def fneq_train(self, rho, u, v, x, y):
        fneq_sum = np.zeros_like(x)
        for i in range(9):
            fneq_part = self.tau**2*self.d2feq_xy2(rho, u, v, x, y, i)
            fneq_sum = np.hstack((fneq_sum, fneq_part))
        fneq_sum = -self.tau * (self.Ex_fneq_(rho, u, v, x, y))+fneq_sum[:, 1:]
        return fneq_sum

    # 得到算例准确解用于误差计算
    def Ex_func(self, x_star, y_star):
        u = self.u_train(x_star, y_star)
        v = self.v_train(x_star, y_star)
        p = self.p_func(x_star)
        rou = p / self.RT
        f_eq = self.f_eq(rou, u, v)
        # excat gradient need to change
        f_neq = self.fneq_train(rou, u, v, x_star, y_star)
        f_neq = tf.cast(f_neq, dtype=tf.float64)
        f_i = f_neq + f_eq
        # tensor change to array
        f_eq = f_eq.eval(session=self.sess)
        f_neq = f_neq.eval(session=self.sess)
        f_i = f_i.eval(session=self.sess)

        return u, v, f_eq, f_neq, f_i

    def Data_Generation(self):
        x_l = self.x_range.min()
        x_u = self.x_range.max()
        y_l = self.y_range.min()
        y_u = self.y_range.max()

        # domain data
        x_data = np.random.random((11000, 1)) - 0.5
        y_data = np.random.random((11000, 1)) * 2 - 0.5
        x_data_1 = np.random.random((6000, 1)) * 1.5 + 0.5
        y_data_1 = np.random.random((6000, 1)) * 2 - 0.5
        X_data = np.vstack((x_data, x_data_1))
        Y_data = np.vstack((y_data, y_data_1))

        # boundary data
        x_1 = (x_u - x_l) * np.random.random((600, 1)) + x_l
        x_2 = (x_u - x_l) * np.random.random((300, 1)) + x_l
        x_3 = np.full((300, 1), -0.5)
        x_4 = np.full((300, 1), 2)
        y_1 = np.full((600, 1), -0.5)
        y_2 = np.full((300, 1), 1.5)
        y_3 = (y_u - y_l) * np.random.random((300, 1)) + y_l
        y_4 = (y_u - y_l) * np.random.random((300, 1)) + y_l
        x_b = np.vstack((x_1, x_2, x_3, x_4))
        y_b = np.vstack((y_1, y_2, y_3, y_4))

        u_data = self.u_train(x_b, y_b)
        v_data = self.v_train(x_b, y_b)
        p_data = self.p_func(x_b)
        rou_data = self.rou_func(p_data)

        return X_data, Y_data, x_b, y_b, u_data, v_data, rou_data
