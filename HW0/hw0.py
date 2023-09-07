import numpy as np
import matplotlib.pyplot as plt


def task3():
    plt.plot([1, 2, 3, 4], [1, 2, 7, 14])
    plt.axis([0, 6, 0, 20])
    plt.show()

def task4():
    x = np.arange(-10,10,0.1)
    y = lambda e: np.sin(e)
    plt.plot(x, [y(i) for i in x])
    plt.show()


if __name__ == "__main__":
    # task3()
    task4()
