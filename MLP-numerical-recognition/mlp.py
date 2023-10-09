import numpy as np


class MLP:
    " Multi-layer perceptron "

    def __init__(self, sizes, beta=1.73, momentum=0.9):

        """
        sizes is a list of length four. The first element is the number of features
                in each samples. In the MNIST dataset, this is 784 (28*28). The second
                and the third  elements are the number of neurons in the first
                and the second hidden layers, respectively. The fourth element is the
                number of neurons in the output layer which is determined by the number
                of classes. For example, if the sizes list is [784, 5, 7, 10], this means
                the first hidden layer has 5 neurons and the second layer has 7 neurons.

        beta is a scalar used in the sigmoid function
        momentum is a scalar used for the gradient descent with momentum
        """

        self.hidden1 = None
        self.hidden2 = None
        self.beta = beta
        self.momentum = momentum

        self.nin = sizes[0]  # number of features in each sample
        self.nhidden1 = sizes[1]  # number of neurons in the first hidden layer
        self.nhidden2 = sizes[2]  # number of neurons in the second hidden layer
        self.nout = sizes[3]  # number of classes / the number of neurons in the output layer

        # Initialise the network of two hidden layers
        self.weights1 = (np.random.rand(self.nin + 1, self.nhidden1) - 0.5) * 2 / np.sqrt(self.nin)  # hidden layer 1
        self.weights2 = (np.random.rand(self.nhidden1 + 1, self.nhidden2) - 0.5) * 2 / np.sqrt(
            self.nhidden1)  # hidden layer 2
        self.weights3 = (np.random.rand(self.nhidden2 + 1, self.nout) - 0.5) * 2 / np.sqrt(
            self.nhidden2)  # output layer

    def train(self, inputs, targets, eta, niterations):
        """
        inputs is a numpy array of shape (num_train, D) containing the training images
                    consisting of num_train samples each of dimension D.

        targets is a numpy array of shape (num_train, D) containing the training labels
                    consisting of num_train samples each of dimension D.

        eta is the learning rate for optimization
        niterations is the number of iterations for updating the weights

        """
        ndata = np.shape(inputs)[0]  # number of data samples
        # adding the bias
        inputs = np.concatenate((inputs, -np.ones((ndata, 1))), axis=1)

        # numpy array to store the update weights
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
        updatew3 = np.zeros((np.shape(self.weights3)))

        for n in range(niterations):

            #############################################################################
            # TODO: implement the training phase of one iteration which consists of two phases:
            # the forward phase and the backward phase. you will implement the forward phase in
            # the self.forwardPass method and return the outputs to self.outputs. Then compute
            # the error (hints: similar to what we did in the lab). Next is to implement the
            # backward phase where you will compute the derivative of the layers and update
            # their weights.
            #############################################################################

            # forward phase
            self.outputs = self.forwardPass(inputs)

            # Error using the sum-of-squares error function
            error = 0.5 * np.sum((self.outputs - targets) ** 2)

            if (np.mod(n, 100) == 0):
                print("Iteration: ", n, " Error: ", error)

            # backward phase
            # Compute the derivative of the output layer. NOTE: you will need to compute the derivative of
            # the softmax function. Hints: equation 4.55 in the book.
            deltao = (-(targets - self.outputs)) * self.outputs * (1 - self.outputs)

            # compute the derivative of the second hidden layer
            deltah2 = self.beta * self.hidden2 * (1.0 - self.hidden2) * np.dot(deltao, np.transpose(self.weights3))

            # compute the derivative of the first hidden layer
            deltah1 = self.beta * self.hidden1 * (1.0 - self.hidden1) * np.dot(deltah2[:, :-1], np.transpose(self.weights2))
            # update the weights of the three layers: self.weights1, self.weights2 and self.weights3
            # here you can update the weights as we did in the week 4 lab (using gradient descent)
            # but you can also add the momentum
            updatew1 = self.momentum * updatew1 + eta * np.dot(np.transpose(inputs), deltah1[:, :-1])
            updatew2 = self.momentum * updatew2 + eta * np.dot(np.transpose(self.hidden1), deltah2[:, :-1])
            updatew3 = self.momentum * updatew3 + eta * np.dot(np.transpose(self.hidden2), deltao)

            self.weights1 -= updatew1
            self.weights2 -= updatew2
            self.weights3 -= updatew3
            #############################################################################
            # END of YOUR CODE
            #############################################################################

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        c = np.exp((-1) * x * self.beta)
        result = 1 / (1 + c)

        return result

    def softmax(self, x):
        x = np.transpose(x)
        c = np.sum(np.exp(x - np.max(x)), axis=0)
        return np.transpose(np.exp(x - np.max(x)) / c)

    def forwardPass(self, inputs):
        """
            inputs is a numpy array of shape (num_train, D) containing the training images
                    consisting of num_train samples each of dimension D.
        """
        #############################################################################
        # TODO: Implement the forward phase of the model. It has two hidden layers
        # and the output layer. The activation function of the two hidden layers is
        # sigmoid function. The output layer activation function is the softmax function
        # because we are working with multi-class classification.
        #############################################################################

        # layer 1
        # compute the forward pass on the first hidden layer with the sigmoid function
        b1 = -np.zeros((np.shape(inputs)[0], 1))
        hidden1in = np.dot(inputs, self.weights1)
        # hidden1in = np.concatenate((hidden1in, b1), axis=1)
        self.hidden1 = self.sigmoid(hidden1in)
        self.hidden1 = np.concatenate((self.hidden1, b1), axis=1)

        # layer 2
        # compute the forward pass on the second hidden layer with the sigmoid function
        b2 = -np.zeros((np.shape(self.hidden1)[0], 1))
        hidden2in = np.dot(self.hidden1, self.weights2)
        # hidden2in = np.concatenate((hidden2in, b2), axis=1)
        self.hidden2 = self.sigmoid(hidden2in)
        self.hidden2 = np.concatenate((self.hidden2, b2), axis=1)

        # output layer
        # compute the forward pass on the output layer with softmax function
        outputsin = np.dot(self.hidden2, self.weights3)
        outputs = self.softmax(outputsin)

        #############################################################################
        # END of YOUR CODE
        #############################################################################
        return outputs

    def evaluate(self, X, y):
        """
            this method is to evaluate our model on unseen samples
            it computes the confusion matrix and the accuracy

            X is a numpy array of shape (num_train, D) containing the testing images
                    consisting of num_train samples each of dimension D.
            y is  a numpy array of shape (num_train, D) containing the testing labels
                    consisting of num_train samples each of dimension D.
        """

        inputs = np.concatenate((X, -np.ones((np.shape(X)[0], 1))), axis=1)
        outputs = self.forwardPass(inputs)
        nclasses = np.shape(y)[1]

        # 1-of-N encoding
        outputs = np.argmax(outputs, 1)
        targets = np.argmax(y, 1)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        print("The confusion matrix is:")
        print(cm)
        print("The accuracy is ", np.trace(cm) / np.sum(cm) * 100)

        return cm
