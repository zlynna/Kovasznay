from DataSet import DataSet
import matplotlib.pyplot as plt
import numpy as np
import os
from sympy import *
import math
import tensorflow as tf

SavePath = './Results/'
if not os.path.exists(SavePath):
    os.makedirs(SavePath)

e = np.array([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
w = np.full((9, 1), 0.0)
w[0] = 4 / 9
w[1:5] = 1 / 9
w[5:] = 1 / 36
RT = 100
Re=10
xi = e * np.sqrt(3 * RT)
u_0 = 0.1581
p_0 = 0.05
L = 1.
nu = u_0/Re
tau = 1.58e-4
lamda = Re / 2 - np.sqrt(Re ** 2 / 4 + 4 * np.pi ** 2)

def plotter(fig, ax, dat, title, xlabel, ylabel, xx, yy):
    dat = np.reshape(dat, (50, 50))
    levels = np.linspace(dat.min(), dat.max(), 20)
    #min = tf.math.reduce_min(dat).eval(session=tf.Session())
    #max = tf.math.reduce_max(dat).eval(session=tf.Session())
    #levels = np.linspace(min, max, 20)
    zs = ax.contourf(xx, yy, dat, cmap='jet', levels=levels)
    fig.colorbar(zs, ax=ax)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return zs

def Eq_res(Eq_res, xx, yy):
    fig5, ((ax34, ax35, ax36), (ax37, ax38, ax39), (ax40, ax41, ax42)) = plt.subplots(3, 3, figsize=(15, 8))
    plotter(fig5, ax34, Eq_res[:, 1], r'$Eq_{f_{0}}$', 'x', 'y', xx, yy)
    plotter(fig5, ax35, Eq_res[:, 1], r'$Eq_{f_{1}}$', 'x', 'y', xx, yy)
    plotter(fig5, ax36, Eq_res[:, 2], r'$Eq_{f_{2}}$', 'x', 'y', xx, yy)
    plotter(fig5, ax37, Eq_res[:, 3], r'$Eq_{f_{3}}$', 'x', 'y', xx, yy)
    plotter(fig5, ax38, Eq_res[:, 4], r'$Eq_{f_{4}}$', 'x', 'y', xx, yy)
    plotter(fig5, ax39, Eq_res[:, 5], r'$Eq_{f_{5}}$', 'x', 'y', xx, yy)
    plotter(fig5, ax40, Eq_res[:, 6], r'$Eq_{f_{6}}$', 'x', 'y', xx, yy)
    plotter(fig5, ax41, Eq_res[:, 7], r'$Eq_{f_{7}}$', 'x', 'y', xx, yy)
    plotter(fig5, ax42, Eq_res[:, 8], r'$Eq_{f_{8}}$', 'x', 'y', xx, yy)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(SavePath + 'Eq_residual.png')

    plt.show()

def grad():
    x = Symbol("x")
    y = Symbol("y")
    t = Symbol("t")
    u = u_0 * (1 - np.exp(lamda * x) * np.cos(2 * np.pi * y))
    v = u_0 * (lamda / (2 * np.pi) * np.exp(lamda * x) * np.sin(2 * np.pi * y))
    rho = (p_0*(1 - 0.5 * np.exp(2 * lamda * x)) + RT)/RT
    u_x = diff(u, x)
    u_y = diff(u, y)
    u_t = diff(u, t)
    u_xx = diff(u_x, x)
    u_yy = diff(u_y, y)
    u_tt = diff(u_t, t)
    u_xt = diff(u_x, t)
    u_xy = diff(u_x, y)
    u_yt = diff(u_y, t)
    v_x = diff(v, x)
    v_y = diff(v, y)
    v_t = diff(v, t)
    v_xx = diff(v_x, x)
    v_yy = diff(v_y, y)
    v_tt = diff(v_t, t)
    v_xt = diff(v_x, t)
    v_xy = diff(v_x, y)
    v_yt = diff(v_y, t)
    r_x = diff(rho, x)
    r_y = diff(rho, y)
    r_t = diff(rho, t)
    r_xx = diff(r_x, x)
    r_yy = diff(r_y, y)
    r_tt = diff(r_t, t)
    r_xt = diff(r_x, t)
    r_xy = diff(r_x, y)
    r_yt = diff(r_y, t)

    u = Symbol("u")
    v = Symbol("v")
    r = Symbol("r")
    w = Symbol("w")
    xiu = Symbol("xiu")
    xiv = Symbol("xiv")
    f = w * r * (1 + (xiu * u + xiv * v) / RT + (xiu * u + xiv * v) ** 2 / 2 / RT ** 2 - (u * u + v * v) / 2 / RT)
    f_u = diff(f, u)
    f_v = diff(f, v)
    f_r = diff(f, r)
    f_uu = diff(f_u, u)
    f_vv = diff(f_v, v)
    f_rr = diff(f_r, r)
    f_tt = f_uu * u_t ** 2 + f_u * u_tt + f_vv * v_t ** 2 + f_v * v_tt + f_rr * r_t ** 2 + f_r * r_tt
    f_xx = f_uu * u_x ** 2 + f_u * u_xx + f_vv * v_x ** 2 + f_v * v_xx + f_rr * r_x ** 2 + f_r * r_xx
    f_yy = f_uu * u_y ** 2 + f_u * u_yy + f_vv * v_y ** 2 + f_v * v_yy + f_rr * r_y ** 2 + f_r * r_yy
    f_xy = f_uu * u_y * u_x + u_xy * f_u + f_rr * r_y * r_x + r_xy * f_r + f_vv * v_y * v_x + v_xy * f_v
    f_xt = f_uu * u_t * u_x + u_xt * f_u + f_rr * r_t * r_x + r_xt * f_r + f_vv * v_t * v_x + v_xt * f_v
    f_yt = f_uu * u_y * u_t + u_yt * f_u + f_rr * r_y * r_t + r_yt * f_r + f_vv * v_y * v_t + v_yt * f_v
    f2 = f_tt+2*xiu*f_xt+2*xiv*f_yt+xiu**2*f_xx+2*xiu*xiv*f_xy+xiv**2*f_yy
    return f2
test=False
if test:
    f2=grad()
    print(f2)
    raise ValueError
def fneq_grad(r, u, v, w, xiu, xiv, x, y, t):
    f2 = xiu**2*(0.22945878296434*r*w*(xiu**2/10000 - 1/100)*np.exp(-6.05969085684496*x)*np.cos(6.28318530717959*y)**2 + 0.0533563232499575*r*w*(xiv**2/10000 - 1/100)*np.exp(-6.05969085684496*x)*np.sin(6.28318530717959*y)**2 - 1.45135220091297*r*w*(-u/100 + xiu*(u*xiu + v*xiv)/10000 + xiu/100)*np.exp(-3.02984542842248*x)*np.cos(6.28318530717959*y) - 0.699863622666408*r*w*(-v/100 + xiv*(u*xiu + v*xiv)/10000 + xiv/100)*np.exp(-3.02984542842248*x)*np.sin(6.28318530717959*y) - 0.00917996332013261*w*(-u**2/200 + u*xiu/100 - v**2/200 + v*xiv/100 + (u*xiu + v*xiv)**2/20000 + 1)*np.exp(-6.05969085684496*x)) + 2*xiu*xiv*(0.47584343418981*r*w*(xiu**2/10000 - 1/100)*np.exp(-6.05969085684496*x)*np.sin(6.28318530717959*y)*np.cos(6.28318530717959*y) - 0.110648438743559*r*w*(xiv**2/10000 - 1/100)*np.exp(-6.05969085684496*x)*np.sin(6.28318530717959*y)*np.cos(6.28318530717959*y) - 3.00976239209241*r*w*(-u/100 + xiu*(u*xiu + v*xiv)/10000 + xiu/100)*np.exp(-3.02984542842248*x)*np.sin(6.28318530717959*y) + 1.45135220091297*r*w*(-v/100 + xiv*(u*xiu + v*xiv)/10000 + xiv/100)*np.exp(-3.02984542842248*x)*np.cos(6.28318530717959*y)) + xiv**2*(0.986787129855652*r*w*(xiu**2/10000 - 1/100)*np.exp(-6.05969085684496*x)*np.sin(6.28318530717959*y)**2 + 0.22945878296434*r*w*(xiv**2/10000 - 1/100)*np.exp(-6.05969085684496*x)*np.cos(6.28318530717959*y)**2 + 6.24153782324891*r*w*(-u/100 + xiu*(u*xiu + v*xiv)/10000 + xiu/100)*np.exp(-3.02984542842248*x)*np.cos(6.28318530717959*y) + 3.00976239209241*r*w*(-v/100 + xiv*(u*xiu + v*xiv)/10000 + xiv/100)*np.exp(-3.02984542842248*x)*np.sin(6.28318530717959*y))
    return f2

y_range = np.array((-0.5, 2))
x_range = np.array((-0.5, 1.5))
t_range = np.array((0., 10.))
Nx = 50
Ny = 50
Nt = 10
N_bc = 300

data = DataSet(x_range, y_range, Nx, Ny, N_bc)
x_test = np.linspace(x_range[0], x_range[1], Nx)
y_test = np.linspace(y_range[0], y_range[1], Ny)
xx, yy = np.meshgrid(x_test, y_test)
x_test = np.ravel(xx).T[:, None]
y_test = np.ravel(yy).T[:, None]
t_test = np.ones_like(x_test)


def u_train(x, y, t):
    u = u_0 * (1 - np.exp(lamda * x) * np.cos(2 * np.pi * y))
    return u

def v_train(x, y, t):
    v = u_0 * (lamda / (2 * np.pi) * np.exp(lamda * x) * np.sin(2 * np.pi * y))
    return v

def r_func(x, y, t):
    r = (p_0*(1 - 0.5 * np.exp(2 * lamda * x)) + RT)/RT
    return r

r = r_func(x_test, y_test, t_test)
u = u_train(x_test, y_test, t_test)
v = v_train(x_test, y_test, t_test)
dfneq_sum = np.zeros_like(x_test)
for k in range(9):
    dfneq = -tau*fneq_grad(r, u, v, w[k],xi[k, 0], xi[k, 1], x_test, y_test, t_test)
    dfneq_sum = np.hstack((dfneq_sum, dfneq))


Eq_res(np.abs(dfneq_sum[:, 1:]), xx, yy)