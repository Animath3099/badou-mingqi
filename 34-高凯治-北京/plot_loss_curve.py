import matplotlib.pyplot as plt


def plot_loss_acc(epoch, loss, acc):
    plt.plot(epoch, loss, color='r', label='loss')        # r表示红色
    plt.plot(epoch, acc, color=(0, 0, 0), label='acc')  # 用RGB值表示颜色
    plt.xlabel('epochs')                                 # x轴表示
    plt.ylabel('loss/acc')                               # y轴表示
    plt.title("train_loss / eval_acc")                   # 图标标题表示
    plt.legend()                                         # 每条折线的label显示
    plt.savefig('test.jpg')                              # 保存图片，路径名为test.jpg
    plt.show()                                           # 显示图片