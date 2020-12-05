import re
import matplotlib.pyplot as plt
import os
from os.path import join


def str2other(strlist, other):
    return list(map(other, strlist))


if __name__ == '__main__':
    for root, dirs, files in os.walk('logs'):
        # print(root, "consumes", end="")
        # print(sum(getsize(join(root, name)) for name in files), end="")
        # print("bytes in", len(files), "non-directory files")
        # if 'CVS' in dirs:
        #     dirs.remove('CVS')  # don't visit CVS directories
        for name in files:
            nameoend = name[:-4]
            file = open(join(root, name), 'r')
            accuracy = []
            epochs = []
            loss = []
            for line in file:
                test_accuracy = re.search(
                    '"test_accuracy": ([0]\.[0-9]+)*', line)
                if test_accuracy:
                    accuracy.append(test_accuracy.group(1))

                epoch = re.search('"epoch": ([0-9]+)*', line)
                if epoch:
                    epochs.append(epoch.group(1))

                train_loss = re.search('"train_loss": ([0-9]\.[0-9]+)*', line)
                if train_loss:
                    loss.append(train_loss.group(1))
            file.close()
            epochs = str2other(epochs, int)
            accuracy = str2other(accuracy, float)
            loss = str2other(loss, float)

            # fig, ax = plt.subplots()
            plt.figure(f'{nameoend}: test_accuracy vs epochs')
            plt.title(f'{nameoend}: test_accuracy vs epochs')
            plt.xlabel('epoch')
            plt.ylabel('test_accuracy')
            # plt.plot(epochs, accuracy, 'b*')
            plt.plot(epochs, accuracy, 'r', label='test_acc')
            plt.plot(epochs, loss, 'y', label='train_loss')
            plt.legend()
            plt.savefig(f'./figure/{nameoend}_test_accuracy.png')

            # plt.figure('train_loss vs epochs')
            # plt.xlabel('epoch')
            # plt.ylabel('train_loss')
            # plt.plot(epochs, loss, 'b*')
            # plt.plot(epochs, loss, 'y')
            # plt.savefig(f'./figure/{nameoend}_train_loss.png')

            # plt.xticks(range(0, 30, 300))
            # plt.yticks([i/100 for i in range(0, 20, 200)])
            # ax.set_xlim(0, 7)
            # ax.set_ylim(0, 2)
            # plt.grid(True)

            # plt.savefig(f'./figure/{nameoend}/train_loss.png')

            # plt.show()
