from DataSet import DataSet
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plotter(fig, ax, dat, title, xlabel, ylabel, xx, yy):
    dat = dat.reshape((100, 125))
    levels = np.linspace(dat.min(), dat.max(), 30)
    zs = ax.contourf(xx, yy, dat, cmap='jet', levels=levels)
    fig.colorbar(zs, ax=ax)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return zs

def relative_error_(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt(np.sum(np.square(pred - exact)) / np.sum(np.square(exact)))
    return tf.sqrt(tf.reduce_sum(tf.square(pred - exact)) / tf.reduce_sum(tf.square(exact)))

def main():
    y_range = np.array((-0.5, 1.5))
    x_range = np.array((-0.5, 2.))
    NX = 17000
    Ny = 17000
    N_bc = 300
    NX_test = 125
    NY_test = 100

    tau = 1.58e-4

    e = np.array([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
    RT = 100
    xi = e * np.sqrt(3 * RT)
    data = DataSet(x_range, y_range, NX, Ny, N_bc)
    x_test = np.linspace(x_range[0], x_range[1], NX_test)
    y_test = np.linspace(y_range[0], y_range[1], NY_test)
    xx, yy = np.meshgrid(x_test, y_test)
    x_test = np.ravel(xx).T[:, None]
    y_test = np.ravel(yy).T[:, None]
    # exact solution
    [u_e, v_e, f_eq_e, f_neq_e, f_i_e] = data.Ex_func(x_test, y_test)

    xi_x = xi[:, 0][:, None]
    xi_y = xi[:, 1][:, None]
    one = np.ones_like(xi_x)
    rou = np.matmul(f_eq_e, one)
    rou_r = np.matmul(f_i_e, one)
    u = np.matmul(f_i_e, xi_x) / rou
    v = np.matmul(f_i_e, xi_y) / rou

    u_error = np.abs(u - u_e)
    v_error = np.abs(v - v_e)
    rou_error = np.abs(rou - rou_r)

    fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 8))
    plotter(fig1, ax1, rou, r'$u$', 'x', 'y', xx, yy)
    plotter(fig1, ax2, u_e, r'$Exact_u$', 'x', 'y', xx, yy)
    plotter(fig1, ax3, u_error, r'$Error_u$', 'x', 'y', xx, yy)
    plotter(fig1, ax4, v, r'$v$', 'x', 'y', xx, yy)
    plotter(fig1, ax5, v_e, r'$Exact_v$', 'x', 'y', xx, yy)
    plotter(fig1, ax6, v_error, r'$Error_v$', 'x', 'y', xx, yy)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

if __name__ == '__main__':
    main()