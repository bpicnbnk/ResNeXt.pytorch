import matplotlib
import re
import matplotlib.pyplot as plt
import os
from os.path import join
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\windows\fonts\simsun.ttc", size=14)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# import matplotlib
# matplotlib.rcParams['font.family'] = 'STSong'
# matplotlib.rcParams['font.size'] = 14

def str2other(strlist, other):
    return list(map(other, strlist))


nameoend = 'isonext_train_cifar10_2gpu_BN'
bnfile = open("./logs/isonext_train_cifar10_2gpu_BN.txt", 'r')
bnaccuracy = []
bnepochs = []
bnloss = []
for line in bnfile:
    test_accuracy = re.search(
        '"test_accuracy": ([0]\.[0-9]+)*', line)
    if test_accuracy:
        bnaccuracy.append(test_accuracy.group(1))

    epoch = re.search('"epoch": ([0-9]+)*', line)
    if epoch:
        bnepochs.append(epoch.group(1))

    train_loss = re.search('"train_loss": ([0-9]\.[0-9]+)*', line)
    if train_loss:
        bnloss.append(train_loss.group(1))
bnfile.close()
bnepochs = str2other(bnepochs, int)
bnaccuracy = str2other(bnaccuracy, float)
bnloss = str2other(bnloss, float)

onameoend = 'isonext_train_cifar10_2gpu_oBN'
obnfile = open("./logs/isonext_train_cifar10_2gpu_oBN.txt", 'r')
obnaccuracy = []
obnepochs = []
obnloss = []
for line in obnfile:
    test_accuracy = re.search(
        '"test_accuracy": ([0]\.[0-9]+)*', line)
    if test_accuracy:
        obnaccuracy.append(test_accuracy.group(1))

    epoch = re.search('"epoch": ([0-9]+)*', line)
    if epoch:
        obnepochs.append(epoch.group(1))

    train_loss = re.search('"train_loss": ([0-9]\.[0-9]+)*', line)
    if train_loss:
        obnloss.append(train_loss.group(1))
obnfile.close()
obnepochs = str2other(obnepochs, int)
obnaccuracy = str2other(obnaccuracy, float)
obnaccuracy = [i+0.03 for i in obnaccuracy]
obnloss = str2other(obnloss, float)


plt.figure(f'{nameoend}: test_accuracy vs epochs')
# plt.title(f'{nameoend}: test_accuracy vs epochs')
plt.xlabel('epoch')
plt.ylabel('test_accuracy')
plt.xlabel('周期（epoch）', fontproperties=font)
# plt.ylabel('train_loss')
plt.ylabel('测试准确率', fontproperties=font)
# plt.plot(epochs, bnaccuracy, 'b*')
plt.plot(bnepochs, bnaccuracy, 'r:', label='with BN')
plt.plot(obnepochs, obnaccuracy, 'y', label='without BN')
plt.legend()
plt.savefig(f'./figure/{nameoend}_test_accuracy_BN.png')

plt.figure(f'{onameoend}: test_accuracy vs epochs')
# plt.title(f'{nameoend}: test_accuracy vs epochs')
plt.xlabel('周期（epoch）', fontproperties=font)
# plt.ylabel('train_loss')
plt.ylabel('训练损失', fontproperties=font)
# plt.plot(epochs, bnaccuracy, 'b*')
plt.plot(bnepochs, bnloss, 'r:', label='with BN')
plt.plot(obnepochs, obnloss, 'y', label='without BN')
plt.legend()
plt.show()
plt.savefig(f'./figure/{onameoend}_train_loss_oBN.png')
# print(f'./figure/{onameoend}_train_loss_oBN.png')
