import numpy as np
import matplotlib.pyplot as plt

class Neuron():
    LEARNING_RATE = 0.0001
    def __init__(self):
        self.weight = np.random.random()
        self.bias = np.random.random()
    
    def forward_prop(self, x, y):
        f = lambda x: self.weight*x + self.bias
        z = np.array([f(x) for x in x])    
        return z
    
    def backward_prop(self, x, y, z):
        sample_size = len(x)
        grad_output = ([2*(z1-y1)/sample_size for z1, y1 in zip(z,y)])
        grad_weight = sum([z1*g1 for z1, g1 in zip(z, grad_output)])
        grad_bias = sum(grad_output)
        return grad_weight, grad_bias
    
    def adjust_parameter(self, gw, gb):
        self.weight = self.weight - gw*self.LEARNING_RATE
        self.bias = self.bias - gb*self.LEARNING_RATE
    
    def train(self, x, y):
        z = self.forward_prop(x, y)
        gw, gb = self.backward_prop(x, y, z)
        self.adjust_parameter(gw, gb)

if __name__ == '__main__':
    f = lambda x: 9*x + 96
    x = np.arange(-50, 50)
    y = np.array([f(x) for x in x])

    neuron = Neuron()
    ITERATION = 5000
    for i in range(ITERATION):
        neuron.train(x, y)
        if i in np.arange(0, ITERATION, ITERATION//20):
            print(f"Training {int(i/(ITERATION)*100)}% completed")

    f1 = lambda x: x*neuron.weight + neuron.bias
    z = np.array([f1(x) for x in x])

    fig, ax = plt.subplots()
    ax.plot(x, y, '-b', label='Actual')
    ax.plot(x, z, '--g', label='Predicted after Training')
    plt.xlabel("x_ axis")
    plt.ylabel("y_axis")
    ax.legend(frameon=False)
    plt.grid()
    plt.show()

    print(f"weight -> {neuron.weight} bias -> {neuron.bias}")
    print("Training is completed")

    x1 = np.random.randint(0, 1000)
    y1 = f(x1)
    y2 = round(neuron.weight * x1 + neuron.bias)
    rate = round((1-(abs(y2-y1)/y1))*100, 2)
    print(f"Actual {y1} predicted {y2} ")
    print(f"{rate} % percentage accurate")
