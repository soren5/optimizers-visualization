from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def landscape_3d(benchmark, optimizers, starting_point=(0, 0)):
    # pyplot settings
    plt.ion()
    fig = plt.figure(figsize=(3, 2), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    params = {'legend.fontsize': 3,
            'legend.handlelength': 3}
    plt.rcParams.update(params)
    plt.axis('off')

    # input (x, y) and output (z) nodes of cost-function graph

    # visualize cost function as a contour plot
    x_val = np.arange(benchmark.domain[0][0], benchmark.domain[0][1], 0.005, dtype=np.float32)
    y_val = np.arange(benchmark.domain[1][0], benchmark.domain[1][1], 0.005, dtype=np.float32)

    x_val_mesh, y_val_mesh = np.meshgrid(x_val, y_val)
    x_val_mesh_flat = x_val_mesh.reshape([-1, 1])
    y_val_mesh_flat = y_val_mesh.reshape([-1, 1])


    z_val_mesh_flat = benchmark.calculate(x_val_mesh_flat, y_val_mesh_flat)
    z_val_mesh = tf.reshape(z_val_mesh_flat, x_val_mesh.shape)
    levels = np.arange(-10, 1, 0.05)

    # ax.contour(x_val_mesh, y_val_mesh, z_val_mesh, levels, alpha=.7, linewidths=0.4)
    # ax.plot_wireframe(x_val_mesh, y_val_mesh, z_val_mesh, alpha=.5, linewidths=0.4, antialiased=True)
    ax.plot_surface(x_val_mesh, y_val_mesh, z_val_mesh.numpy(), alpha=.4, cmap=cm.coolwarm)
    plt.draw()

    # starting location for variables
    x_i = starting_point[0]
    y_i = starting_point[1]

    # create variable pair (x, y) for each optimizer
    x_var, y_var = [], []
    cost = []
    for i in range(len(optimizers)):
        x_var.append(tf.Variable(x_i, [1], dtype=tf.float32))
        y_var.append(tf.Variable(y_i, [1], dtype=tf.float32))
        cost.append(benchmark.calculate(x_var[i], y_var[i]))

    # define method of gradient descent for each graph
    # optimizer label name, learning rate, color
    
    #ops_param = [opt.name for opt in optimizers]
    ops_param = [
        ['SGD', 'r'],
        ['Adam', 'c'],
        ] #TODO THIS IS A PLACEHOLDER


    # 3d plot camera zoom, angle
    xlm = ax.get_xlim3d()
    ylm = ax.get_ylim3d()
    zlm = ax.get_zlim3d()
    ax.set_xlim3d(xlm[0] * 0.5, xlm[1] * 0.5)
    ax.set_ylim3d(ylm[0] * 0.5, ylm[1] * 0.5)
    ax.set_zlim3d(zlm[0] * 0.5, zlm[1] * 0.5)
    azm = ax.azim
    ele = ax.elev + 40
    ax.view_init(elev=ele, azim=azm)





    # use last location to draw a line to the current location
    last_x, last_y, last_z = [], [], []
    plot_cache = [None for _ in range(len(optimizers))]

    # loop each step of the optimization algorithm
    steps = 1000
    for iter in range(steps):
        for i, op in enumerate(optimizers):
            # run a step of optimization and collect new x and y variable values

            with tf.GradientTape() as tape:
                z_var = benchmark.calculate(x_var[i], y_var[i])
                var_list = [x_var[i], y_var[i]]
                grads = (tape.gradient(z_var, var_list))
                op.apply_gradients(zip(grads, var_list))

            x_val, y_val, z_val = x_var[i].numpy(), y_var[i].numpy(), z_var.numpy()
            print(ops_param[i], x_val, y_val, z_val)
            # move dot to the current value
            if plot_cache[i]:
                plot_cache[i].remove()
            plot_cache[i] = ax.scatter(x_val, y_val, z_val, s=3, depthshade=True, label=ops_param[i][0], color=ops_param[i][1])

            # draw a line from the previous value
            if iter == 0:
                last_z.append(z_val)
                last_x.append(x_i)
                last_y.append(y_i)

            ax.plot([last_x[i], x_val], [last_y[i], y_val], [last_z[i], z_val], linewidth=0.5, color=ops_param[i][1])
            last_x[i] = x_val
            last_y[i] = y_val
            last_z[i] = z_val

        if iter == 0:
            legend = [x[0] for x in ops_param]
            plt.legend(plot_cache, legend)

        plt.savefig('figures/' + str(iter) + '.png')
        print('iteration: {}'.format(iter))
        plt.pause(0.0001)