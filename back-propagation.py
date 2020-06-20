import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(s):
    return s * (1 - s)


class NeuralNetwork:

    def __init__(self, X, Y):
        self.inputs = X  # Each row is a training example, each column is a feature [X1,..., X7] (10x7)
        self.Y = Y  # The real output data (10x4)
        self.hidden_nodes = 4  # Number of nodes in the hidden layer
        self.weights1 = np.random.rand(self.inputs.shape[1], self.hidden_nodes)  # (7x4)
        self.weights2 = np.random.rand(self.Y.shape[1], self.hidden_nodes)  # (4x4)
        self.outputs = np.zeros(Y.shape)  # The net's output data (10x4)

        self.error_history = []
        self.epoch_list = []

    def feed_forward(self):
        self.hidden = sigmoid(np.dot(self.inputs, self.weights1))
        self.outputs = sigmoid(np.dot(self.hidden, self.weights2))

    def backpropagation(self):
        self.errors = self.Y - self.outputs
        derivatives2 = sigmoid_derivative(self.outputs)
        deltas2 = self.errors * derivatives2
        self.weights2 += np.dot(self.hidden.T, deltas2)

        derivatives1 = sigmoid_derivative(self.hidden)
        deltas1 = np.dot(deltas2, self.weights2.T) * derivatives1
        self.weights1 += np.dot(self.inputs.T, deltas1)

    def train(self, epochs=20000):
        print(f"Training network ({epochs} epochs)...\n\n")
        for epoch in range(epochs):
            self.feed_forward()
            self.backpropagation()

            self.error_history.append(np.average(np.abs((self.errors))))
            self.epoch_list.append(epoch)

    def predict(self, new_input):
        # An additional feed_forward-like to make a prediction
        hidden = sigmoid(np.dot(new_input, self.weights1))
        prediction = sigmoid(np.dot(hidden, self.weights2))
        return np.round(prediction, 3)


def main():
    inputs = np.array([[1, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 0, 0, 0, 0],
                       [1, 1, 0, 1, 1, 0, 1],
                       [1, 1, 1, 1, 0, 0, 1],
                       [0, 1, 1, 0, 0, 1, 1],
                       [1, 0, 1, 1, 0, 1, 1],
                       [1, 0, 1, 1, 1, 1, 1],
                       [1, 1, 1, 0, 0, 0, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 0, 1, 1]])

    outputs = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 1],
                        [0, 1, 1, 0],
                        [0, 1, 1, 1],
                        [1, 0, 0, 0],
                        [1, 0, 0, 1]])

    NN = NeuralNetwork(inputs, outputs)
    NN.train()

    # Plot the error over the entire training duration
    plt.figure(figsize=(10, 5))
    plt.plot(NN.epoch_list, NN.error_history)
    plt.title('Error percentage during training')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show(block=False)

    counter = 0
    while counter < 10:
        try:
            test = int(input("Ingrese el digito el cual quiere que la red prediga\n"
                             "(este sera entregado a la red en formato de 7 segmentos): "))
            if 0 <= test <= 9:
                segments = inputs[test]
                prediction = NN.predict(segments)
                expected = outputs[test]
                counter += 1
                print(f"\tValor ingresado: {segments}\n\tValor predicho: {prediction}\n\tValor correcto: {expected}\n")
            else:
                print("Debe ingresar un solo digito")
        except ValueError:
            print("Debe ingresar un digito")
            continue


if __name__ == "__main__":
    main()