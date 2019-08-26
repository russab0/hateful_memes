import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

smooth_param = 0.95

train_mses = []
valid_mses = []

n_mses = 0
n_valid = 0

def smooth(list, smooth_param):
    value = list[0]
    list_smooth = [value]
    for x in list[1:]:
        value = value * smooth_param + x * (1-smooth_param)
        list_smooth.append(value)

    return list_smooth
if __name__ == '__main__':

    for e in tf.train.summary_iterator("H100x4_IF4098v2jewsValidLoss/events.out.tfevents.1556810569.c9"):
        for v in e.summary.value:
            if v.tag == "train/mse":
                train_mses.append(v.simple_value)
                n_mses += 1

            elif v.tag == "validation/valid_mse":
                valid_mses.append(v.simple_value)
                n_valid += 1

    batches_per_epoch = n_mses/n_valid

    mse_train_smooth = smooth(train_mses, smooth_param)
    mse_valid_smooth = smooth(valid_mses, smooth_param)

    mse_train_smooth = np.array(mse_train_smooth)
    mse_valid_smooth = np.array(mse_valid_smooth)

    train_steps = np.arange(0, n_valid, step=n_valid/n_mses)
    valid_steps = np.arange(0, n_valid, 1)
    valid_steps += 1

    # fig, ax1 = plt.subplots()

    # color = 'tab:blue'
    # ax1.set_xlabel('Number of epoch')
    # ax1.set_ylabel('Train', color=color)
    # ax1.plot(train_steps, mse_train_smooth, color=color)
    # ax1.tick_params(axis='MSE', labelcolor=color)
    #
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #
    # color = 'tab:orange'
    # ax2.set_ylabel('Validation',
    #                color=color)  # we already handled the x-label with ax1
    # ax2.plot(valid_steps, mse_valid_smooth, color=color)
    # ax2.tick_params(axis='MSE', labelcolor=color)
    #
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #
    #
    # plt.show()


    plt.plot(train_steps, mse_train_smooth)
    plt.plot(valid_steps, mse_valid_smooth)
    plt.ylabel("MSE")
    plt.xlabel("Number of epoch")
    plt.show()



print("mse", n_mses)
print("valid", n_valid)