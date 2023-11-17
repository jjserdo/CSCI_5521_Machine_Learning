import numpy as np
import warnings

def preprocess_data(data, mean=None, std=None):
    # for this function, you will need to normalize the data, to make it have zero mean and unit variance
    # to avoid the numerical issue, we can add 1e-15 to the std

    # it has different process in train set, and validation/test set
    if mean is not None or std is not None:
        # mean and std is precomputed with the training data
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        ''' Start Code '''
        data = (data - mean)/std
        ''' End Code '''
        # ------------------------------------------------------------------------------------------------------------

        return data
    else:
        # compute the mean and std based on the training data
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        ''' Start Code '''
        mean = np.mean(data, axis=0)
        std  = np.std(data, axis=0) + 1E-15
        data = (data - mean) / std
        ''' End Code '''
        # ------------------------------------------------------------------------------------------------------------

        return data, mean, std

def preprocess_label(label):
    # to handle the loss function computation, convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label), 10])
    # ------------------------------------------------------------------------------------------------------------
    # complete your code here
    ''' Start Code '''
    for i in range(one_hot.shape[0]):
        one_hot[i][label[i]] = 1
    ''' End Code '''
    # ------------------------------------------------------------------------------------------------------------

    return one_hot

def sigmoid(x):
    # implement the sigmoid activation function for hidden layer
    # ------------------------------------------------------------------------------------------------------------
    # complete your code here
    ''' Start Code '''
    f_x = 1 / (1 + np.exp(-x))
    ''' End Code '''
    # ------------------------------------------------------------------------------------------------------------
    return f_x

def dsigmoid(x):
    # implement the derivative of sigmoid
    ''' Start Code '''
    df_x = sigmoid(x) * (1 - sigmoid(x))
    ''' End Code '''
    return df_x

def Relu(x):
    # implement the Relu activation function for hidden layer
    # ------------------------------------------------------------------------------------------------------------
    # complete your code here
    ''' Start Code '''
    f_x = np.maximum(0, x)
    ''' End Code '''
    # ------------------------------------------------------------------------------------------------------------
    return f_x

def dRelu(x):
    df_x = np.zeros((x.shape[0],x.shape[1]))
    for i in range(df_x.shape[0]):
        for j in range(df_x.shape[1]):
            if x[i][j] > 0:
                df_x[i][j] = 1
            else:
                df_x[i][j] = 0
    return df_x

def tanh(x):
    # implement the tanh activation function for hidden layer
    # ------------------------------------------------------------------------------------------------------------
    # complete your code here
    ''' Start Code '''
    f_x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    ''' End Code '''
    # ------------------------------------------------------------------------------------------------------------

    return f_x

def dtanh(x):
    df_x = 1 - tanh(x) ** 2
    return df_x

def softmax(x):
    # implement the softmax activation function for output layer
    ''' Start Code '''
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    f_x = e_x / np.sum(e_x, axis=-1, keepdims=True)
    ''' End Code '''
    return f_x

def dsoftmax(x):
    df_x = softmax(x) * (1 - softmax(x))
    return df_x

def crossE(y, yhat):
    # implement the cross-entropy loss
    yhat = np.clip(yhat, 1E-15, 1 - 1E-15)
    E = - y * np.log(yhat) - (1 - y) * np.log(1 - yhat)
    return E

def dcrossE(y, yhat):
    # implement the cross-entropy loss
    yhat = np.clip(yhat, 1E-15, 1 - 1E-15)
    dE = - (y / yhat) + (1 - y) / (1 - yhat)
    return dE

class MLP:
    def __init__(self, num_hid, activation="Relu"):
        # initialize the weights
        np.random.seed(2022)
        
        self.weight_1 = np.random.random([64, num_hid]) / 100
        self.bias_1 = np.random.random([1, num_hid]) / 100
        self.weight_2 = np.random.random([num_hid, 10]) / 100
        self.bias_2 = np.random.random([1, 10]) / 100
        
        # note that in your implementation, you need to consider your selected activation.
        self.activation = activation

    def fit(self, train_x, train_y, valid_x, valid_y):
        # initialize learning rate
        lr = 5e-4
        # initialize the counter of recording the number of epochs that the model does not improve
        # and log the best validation accuracy
        count = 0
        best_valid_acc = 0

        """
        You also need to stopping criteria the training if we find no improvement over the best_valid_acc for more than 100 iterations
        """
        while count <= 100:
            # in this case, you will train all the samples (full-batch gradient descents)
            # implement the forward pass,
            # you also need to consider the specific selected activation function on hidden layer
            # ------------------------------------------------------------------------------------------------------------
            # complete your code here
            ''' Start Code '''
            np.seterr(invalid='ignore')
            warnings.filterwarnings('ignore')
            a1 = np.dot(train_x, self.weight_1) + self.bias_1 
            # (1000,64)(64,5) + (1,5) ---> (1000,5)
            
            if self.activation == 'Sigmoid':
                a2 = sigmoid(a1)
            if self.activation == 'Relu':
                a2 = Relu(a1)
            if self.activation == 'tanh':
                a2 = tanh(a1)
                
            a3  = np.dot(a2, self.weight_2) + self.bias_2 
            # (1000,5)(5,10) -> 1000, 10
            yhat = softmax(a3)
            #L = crossE(train_y.T, yhat)
            ''' End Code '''
            # ------------------------------------------------------------------------------------------------------------

            # implement the backward pass (also called the backpropagation)
            # compute the gradients for different parameters, e.g. self.weight_1, self.bias_1, self.weight_2, self.bias_2
            # you also need to consider the specific selected activation function on hidden layer
            # ------------------------------------------------------------------------------------------------------------
            # complete your code here
            
            dLda3 = dcrossE(train_y, yhat) * dsoftmax(a3)
            dLdw2 = np.dot(a2.T, dLda3)
            dLdb2 = np.sum(dLda3, axis=0, keepdims=True) 
            dLda2 = np.dot(dLda3, self.weight_2.T)
            
            if self.activation == 'Sigmoid':
                dLda1 = dLda2 * dsigmoid(a1)
            if self.activation == 'Relu':
                dLda1 = dLda2 * dRelu(a1)
            if self.activation == 'tanh':
                dLda1 = dLda2 * dtanh(a1)
           
            dLdw1 = np.dot(train_x.T, dLda1) 
            dLdb1 = np.sum(dLda1, axis=0, keepdims=True) 
        
            # ------------------------------------------------------------------------------------------------------------

            # update the corresponding parameters based on sum of gradients for above the training samples
            # ------------------------------------------------------------------------------------------------------------
            # complete your code here
            ''' Start Code '''
            self.weight_1 -= lr * dLdw1
            self.weight_2 -= lr * dLdw2
            self.bias_1 -= lr * dLdb1
            self.bias_2 -= lr * dLdb2
            ''' End Code '''
            # ------------------------------------------------------------------------------------------------------------

            # evaluate the accuracy on the validation data
            predictions = self.predict(valid_x)
            cur_valid_acc = (predictions.reshape(-1) == valid_y.reshape(-1)).sum() / len(valid_x)

            # compare the current validation accuracy, if cur_valid_acc > best_valid_acc, we will increase count by it
            if cur_valid_acc > best_valid_acc:
                best_valid_acc = cur_valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self, x):
        # go through the MLP and then obtain the probability of each category
        # you also need to consider the specific selected activation function on hidden layer
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        ''' Start Code '''
        a1 = np.dot(x, self.weight_1) + self.bias_1 
        # (1000,64)(64,5) + (1,5) ---> (1000,5)
        
        if self.activation == 'Sigmoid':
            a2 = sigmoid(a1)
        if self.activation == 'Relu':
            a2 = Relu(a1)
        if self.activation == 'tanh':
            a2 = tanh(a1)
            
        a3  = np.dot(a2, self.weight_2) + self.bias_2 
        # (1000,5)(5,10) -> 1000, 10
        yhat = softmax(a3)
        ''' End Code '''
        # ------------------------------------------------------------------------------------------------------------

        # convert category probability to the corresponding predicted labels
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        ''' Start Code '''
        y1 = np.argmax(yhat, axis=1)
        ''' End Code '''
        # ------------------------------------------------------------------------------------------------------------

        return y1

    def get_hidden(self, x):
        # obtain the hidden layer features, the one after applying activation function
        # you also need to consider the specific selected activation function on hidden layer
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        ''' Start Code '''
        a1 = np.dot(x, self.weight_1) + self.bias_1 
        # (1000,64)(64,5) + (1,5) ---> (1000,5)
        
        if self.activation == 'Sigmoid':
            a2 = sigmoid(a1)
        if self.activation == 'Relu':
            a2 = Relu(a1)
        if self.activation == 'tanh':
            a2 = tanh(a1)
        ''' End Code '''
        # ------------------------------------------------------------------------------------------------------------

        return a2

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
