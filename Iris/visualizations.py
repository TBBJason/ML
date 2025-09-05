import matplotlib
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt

class visualization():
    def __init__(self, dataset):
        self.dataset = dataset
    def print_data(self):
        print("shape: ", self.dataset.shape)

        print("head: \n", self.dataset.head(20))

        print("distribution: ", self.dataset.groupby('class').size())

    def data_visualization(self):
        print("univariate box plot:")
        self.dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
        # plt.show()

        print("multivariate scatterplot")
        scatter_matrix(self.dataset)
        plt.show()

