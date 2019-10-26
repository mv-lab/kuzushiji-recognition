import matplotlib.pyplot as plt


def plot_loss(log_df):
    loss_trn = log_df['loss_trn'].values
    less_val = log_df['loss_val'].values
    plt.plot(loss_trn, label='train')
    plt.plot(less_val, label='valid')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('loss history')
    plt.grid()
    plt.show();
    
    
def plot_score(log_df, col1='trn_score',col2='val_score'):
    trn_score = log_df[col1].values
    val_score = log_df[col2].values
    plt.plot(trn_score, label='train')
    plt.plot(val_score, label='valid')
    plt.legend()
    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.title('score history')
    plt.grid()
    plt.show();
    
    
def show_val_predictions(img, x, y):
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1, 3, 1, xticks=[], yticks=[])
    plt.imshow(img.numpy().transpose(1,2,0))
    ax.set_title('img')
    
    ax = fig.add_subplot(1, 3, 2, xticks=[], yticks=[])
    plt.imshow(x.numpy())
    ax.set_title('pred')
    
    ax = fig.add_subplot(1, 3, 3, xticks=[], yticks=[])
    plt.imshow(y.numpy())
    ax.set_title('truth')

    plt.show()
    plt.close()