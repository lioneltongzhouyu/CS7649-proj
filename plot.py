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


parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--number", type=str)
args = parser.parse_args()

#pth = args.pth

def getFlist(path):
    for root, dirs, files in os.walk(path):
        print('files:', files)     #文件名称，返回list类型
    return files

file_path = "./output/" + args.algorithm + "/" + args.dataset + "/" + args.number + "/summary/"
file_name = getFlist(file_path)
print(file_name)
file_path += file_name[0]

cnt = 0
plot_list = []
for event in tf.train.summary_iterator(
        file_path):
    cnt += 1
    # print(len(event.summary.value))
    for value in event.summary.value:
        plot_list.append(value.histo.sum)
x = []
for i in range(len(plot_list)):
    x.append(i + 1)

plt.plot(x, plot_list)
plt.legend()

plt.xlabel('Epochs')
plt.ylabel('Performance')
# plt.title("Pretrained on ")

output_path = "./output/pic/" +  args.algorithm + "_" + args.dataset + "_" + args.number + ".png"
plt.savefig(output_path)

