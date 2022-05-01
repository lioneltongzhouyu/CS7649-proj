import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

cnt = 0
plot_list = []
for event in tf.train.summary_iterator(
        "./output/summary/gail_ant3"):
    cnt += 1
    # print(len(event.summary.value))
    for value in event.summary.value:
        plot_list.append(value.histo.sum)
x = []
for i in range(len(plot_list)):
    x.append(i + 1)

plt.plot(x, plot_list)
plt.legend()
plt.show()