import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def start_tsne(x_train, y_train):
    print("initiating data visualization...")
    X_tsne = TSNE().fit_transform(x_train)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train)
    plt.colorbar()
    plt.show()


def start_tsne1(x_train, y_train):
    print("initiating data visualization...")
    X_tsne = TSNE().fit_transform(x_train)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train)
    font_dict = dict(fontsize=14,
                     family='TimesNewRoman',
                     weight='normal')
    plt.xlabel('Dimension1', fontdict=font_dict)
    plt.ylabel('Dimension2', fontdict=font_dict)
    plt.colorbar()
    plt.show()
