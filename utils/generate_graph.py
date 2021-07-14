import matplotlib.pyplot as plt

def loss_vs_epoch(avg_valid_losses, avg_train_losses):
    fig = plt.figure()
    ######1st plot#########
    ax1 = fig.add_subplot()
    ax1.set_ylabel('valid  /  train loss')
    ax1.set_xlabel('number of epochs')
    halt = avg_valid_losses.index(min(avg_valid_losses))


    plt.axvline(x=halt, color='r', linestyle="--", label="stop training")

    print(avg_valid_losses.index(min(avg_valid_losses)))


    plt.plot(list(range(len(avg_valid_losses))), avg_valid_losses, label = "valid loss")
    plt.plot(list(range(len(avg_valid_losses))), avg_train_losses, label="train loss")

    h,labels = ax1.get_legend_handles_labels()
    labels[:1] = ['stop training','valid loss', 'train loss',]
    ax1.legend(labels=labels)
    plt.show()