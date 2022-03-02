import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from DataSet import DataSet

SavePath = './Results/'
if not os.path.exists(SavePath):
    os.makedirs(SavePath)

class Plotting:
    def __init__(self, x_range, NX, y_range, NY, sess):
        self.x_range = x_range
        self.y_range = y_range
        self.NX = NX
        self.NY = NY
        self.N_bc = 300
        self.sess = sess

    def Saveplot(self, u_pre, v_pre, fneq_pre, feq_pre, Eq_res, x_train, y_train):
        # exact solution
        data = DataSet(self.x_range, self.y_range, self.NX, self.NY, self.N_bc)
        x_test = np.linspace(self.x_range[0], self.x_range[1], self.NX)
        y_test = np.linspace(self.y_range[0], self.y_range[1], self.NY)
        xx, yy = np.meshgrid(x_test, y_test)
        x_test = np.ravel(xx).T[:, None]
        y_test = np.ravel(yy).T[:, None]
        # exact solution
        [u_e, v_e, f_eq_e, f_neq_e, f_i_e] = data.Ex_func(x_test, y_test)
        tf_dict = {x_train: x_test, y_train: y_test}

        """u = self.sess.run(u_pre, tf_dict)
        v = self.sess.run(v_pre, tf_dict)"""
        f_neq = self.sess.run(fneq_pre, tf_dict)
        Eq_res = self.sess.run(Eq_res, tf_dict)
        f_eq = self.sess.run(feq_pre, tf_dict)

        e = np.array([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
        RT = 100
        xi = e * np.sqrt(3 * RT)
        tau = 1.58e-4

        xi_x = xi[:, 0][:, None]
        xi_y = xi[:, 1][:, None]
        one = np.ones_like(xi_x)
        rou = np.matmul(f_eq, one)
        rou_r = np.matmul(f_i_e, one)
        u = np.matmul(f_eq, xi_x) / rou
        v = np.matmul(f_eq, xi_y) / rou

        u_error = np.abs(u - u_e)
        v_error = np.abs(v - v_e)
        f_neq_error = np.abs(f_neq - f_neq_e)
        feq_error = np.abs(f_eq - f_eq_e)

        for i in range(1, 9):
            error_fneq = self.relative_error_(f_neq[:, i], f_neq_e[:, i])
            print('Error f_neq %o: %e' % (i, error_fneq))

        for i in range(9):
            error_eq = np.mean(Eq_res[:, i])
            print('Error Equation_res %o: %e' % (i, error_eq))

        error_u = self.relative_error_(u, u_e)
        error_v = self.relative_error_(v, v_e)
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))

        fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 8))
        self.plotter(fig1, ax1, u, r'$PINN-BGK_u$', 'x', 'y', xx, yy)
        self.plotter(fig1, ax2, u_e, r'$Exact_u$', 'x', 'y', xx, yy)
        self.plotter(fig1, ax3, u_error, r'$Error_u$', 'x', 'y', xx, yy)
        self.plotter(fig1, ax4, v, r'$PINN-BGK_v$', 'x', 'y', xx, yy)
        self.plotter(fig1, ax5, v_e, r'$Exact_v$', 'x', 'y', xx, yy)
        self.plotter(fig1, ax6, v_error, r'$Error_v$', 'x', 'y', xx, yy)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.savefig(SavePath + 'speed.png')

        fig2, ((ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(3, 3, figsize=(15, 8))
        self.plotter(fig2, ax7, f_eq[:, 1], r'$PINN-BGK_{f_{0}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig2, ax8, f_eq_e[:, 1], r'$Exact_{f_{0}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig2, ax9, feq_error[:, 1], r'$Error_{f_{0}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig2, ax10, f_neq[:, 1], r'$PINN-BGK_{f_{1}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig2, ax11, f_neq_e[:, 1], r'$Exact_{f_{1}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig2, ax12, f_neq_error[:, 1], r'$Error_{f_{1}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig2, ax13, f_neq[:, 2], r'$PINN-BGK_{f_{2}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig2, ax14, f_neq_e[:, 2], r'$Exact_{f_{2}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig2, ax15, f_neq_error[:, 2], r'$Error_{f_{2}^{neq}}$', 'x', 'y', xx, yy)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(SavePath + 'f_neq1-2.png')

        fig3, ((ax16, ax17, ax18), (ax19, ax20, ax21), (ax22, ax23, ax24)) = plt.subplots(3, 3, figsize=(15, 8))
        self.plotter(fig3, ax16, f_neq[:, 3], r'$PINN-BGK_{f_{3}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig3, ax17, f_neq_e[:, 3], r'$Exact_{f_{3}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig3, ax18, f_neq_error[:, 3], r'$Error_{f_{3}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig3, ax19, f_neq[:, 4], r'$PINN-BGK_{f_{4}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig3, ax20, f_neq_e[:, 4], r'$Exact_{f_{4}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig3, ax21, f_neq_error[:, 4], r'$Error_{f_{4}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig3, ax22, f_neq[:, 5], r'$PINN-BGK_{f_{5}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig3, ax23, f_neq_e[:, 5], r'$Exact_{f_{5}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig3, ax24, f_neq_error[:, 5], r'$Error_{f_{5}^{neq}}$', 'x', 'y', xx, yy)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(SavePath + 'f_neq3-5.png')

        fig4, ((ax25, ax26, ax27), (ax28, ax29, ax30), (ax31, ax32, ax33)) = plt.subplots(3, 3, figsize=(15, 8))
        self.plotter(fig4, ax25, f_neq[:, 6], r'$PINN-BGK_{f_{6}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig4, ax26, f_neq_e[:, 6], r'$Exact_{f_{6}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig4, ax27, f_neq_error[:, 6], r'$Error_{f_{6}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig4, ax28, f_neq[:, 7], r'$PINN-BGK_{f_{7}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig4, ax29, f_neq_e[:, 7], r'$Exact_{f_{7}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig4, ax30, f_neq_error[:, 7], r'$Error_{f_{7}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig4, ax31, f_neq[:, 8], r'$PINN-BGK_{f_{8}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig4, ax32, f_neq_e[:, 8], r'$Exact_{f_{8}^{neq}}$', 'x', 'y', xx, yy)
        self.plotter(fig4, ax33, f_neq_error[:, 8], r'$Error_{f_{8}^{neq}}$', 'x', 'y', xx, yy)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(SavePath + 'f_neq6-8.png')

        fig5, ((ax34, ax35, ax36), (ax37, ax38, ax39), (ax40, ax41, ax42)) = plt.subplots(3, 3, figsize=(15, 8))
        self.plotter(fig5, ax34, Eq_res[:, 1], r'$Eq_{f_{0}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax35, Eq_res[:, 1], r'$Eq_{f_{1}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax36, Eq_res[:, 2], r'$Eq_{f_{2}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax37, Eq_res[:, 3], r'$Eq_{f_{3}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax38, Eq_res[:, 4], r'$Eq_{f_{4}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax39, Eq_res[:, 5], r'$Eq_{f_{5}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax40, Eq_res[:, 6], r'$Eq_{f_{6}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax41, Eq_res[:, 7], r'$Eq_{f_{7}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax42, Eq_res[:, 8], r'$Eq_{f_{8}}$', 'x', 'y', xx, yy)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(SavePath + 'Eq_residual.png')

        plt.show()

    def plotter(self, fig, ax, dat, title, xlabel, ylabel, xx, yy):
        dat = dat.reshape((100, 125))
        levels = np.linspace(dat.min(), dat.max(), 20)
        zs = ax.contourf(xx, yy, dat, cmap='jet', levels=levels)
        fig.colorbar(zs, ax=ax)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        return zs

    def relative_error_(self, pred, exact):
        if type(pred) is np.ndarray:
            return np.sqrt(np.sum(np.square(pred - exact)) / np.sum(np.square(exact)))
        return tf.sqrt(tf.reduce_sum(tf.square(pred - exact)) / tf.reduce_sum(tf.square(exact)))
