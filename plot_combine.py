import imp
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

warnings.filterwarnings('ignore')

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

sns.set_theme()


#pth = args.pth

def getmap(algorithm, dataset, number):
    def getFlist(path):
        for root, dirs, files in os.walk(path):
            print('files:', files)  # 文件名称，返回list类型
        return files

    file_path = "./output/" + algorithm + "/" + dataset + "/" + number + "/summary/"
    file_name = getFlist(file_path)
    file_path += file_name[0]
    cnt = 0
    plot_list = []
    for event in tf.train.summary_iterator(
            file_path):
        cnt += 1
        for value in event.summary.value:
            plot_list.append(value.histo.sum)
    x = []
    for i in range(len(plot_list)):
        x.append(i + 1)
    return x, plot_list


plt.figure(figsize=(10, 8), dpi = 80)

x, plot_list = getmap("basic_gail", "Antorigin-v0", "3")
plt.plot(x[:68], plot_list[:68], label="baseline, default Ant-v0")

x, plot_list = getmap("rl_gail", "Antorigin-v0", "2")
plt.plot(x[:68], plot_list[:68], label="rl-gail, default Ant-v0")

x, plot_list = getmap("bc_gail", "Antorigin-v0", "1")
plt.plot(x[:68], plot_list[:68], label="bc-gail, default Ant-v0")
x, plot_list = getmap("bc_gail", "AntLeg2-v0", "3")
plt.plot(x[:68], plot_list[:68], label="bc-gail, Ant-v0 with 50% leg length")
x, plot_list = getmap("bc_gail", "AntLeg3-v0", "2")
plt.plot(x[:68], plot_list[:68], label="bc-gail, Ant-v0 with 75% leg length")
x, plot_list = getmap("bc_bc_gail", "AntLeg2-v0", "1")
plt.plot(x[:68], plot_list[:68], label="bc-gail, Ant-v0 with repeated dynamices")

x, plot_list = getmap("gail_gail", "AntLeg2-v0", "1")
plt.plot(x[:68], plot_list[:68], label="gail-gail, Ant-v0 with 50% leg length")
x, plot_list = getmap("gail_gail", "AntLeg3-v0", "3")
plt.plot(x[:68], plot_list[:68], label="gail-gail, Ant-v0 with 75% leg length")
x, plot_list = getmap("gail_gail_gail", "AntLeg2-v0", "1")
plt.plot(x[:68], plot_list[:68], label="gail-gail, Ant-v0 with repeated dynamices")


plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Performance')
plt.yticks(range(-1000, 1900, 200))

output_path = "output/pic_combine/c4.png"
plt.savefig(output_path)

