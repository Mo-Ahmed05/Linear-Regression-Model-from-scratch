import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, epsilon=1e-12, lamda=0, regularization=None):
        self.lr = learning_rate
        self.iterations = iterations
        self.epsilon = epsilon

        self.lamda = lamda
        self.reg = regularization

        self.weights = None
        self.bias = 0
        self.scaled = False
        self.min_max_scale = None
        
        self.cost_history = []

    def min_normalization(self, x):
        min_vals = np.min(x, axis=0)
        max_vals = np.max(x, axis=0)

        self.scaled = True
        self.min_max_scale = (min_vals, max_vals)

        return (x - min_vals) / (max_vals - min_vals)
    
    def get_regularization_gradient(self, m):
            if self.reg == "lasso":
                return (self.lamda / (2 * m)) * np.sign(self.weights)
            elif self.reg == "ridge":
                return (self.lamda / m) * self.weights
            else:
                return 0
            
    def get_regularization_cost(self, m):
            if self.reg == "lasso":
                return (self.lamda / (2*m)) * np.sum(np.abs(self.weights))
            elif self.reg == "ridge":
                return (self.lamda / (2*m)) * np.sum(self.weights**2)
            else:
                return 0

    def fit(self, x, y):

        # m: number of rows (samples), n: number of columns (features)
        m, n = x.shape
        self.weights = np.zeros(n)
        self.bias = 0
        self.cost_history = []  # Reset history on new fit

        for i in range(self.iterations):
            y_predicted = np.dot(x, self.weights) + self.bias
            loss = y_predicted - y

            # standard gradient
            d_RSS = (1/m) * np.dot(x.T, loss)
            
            # regularized gradient
            d_regularized = self.get_regularization_gradient(m)

            # The derivatives
            dw = d_RSS + d_regularized
            db = (1/m) * np.sum(loss)
            
            # Update the parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Calculate total cost based on reg type
            mse_cost = (1/(2*m)) * np.sum(loss ** 2)
            reg_cost = self.get_regularization_cost(m)
            total_cost = mse_cost + reg_cost
            
            self.cost_history.append(total_cost)

            # Print cost updates
            if i % (self.iterations // 10) == 0:
                print(f'iteration: {i}, cost: {total_cost}')

            # Stop condition
            if i > 0 and abs(self.cost_history[i] - self.cost_history[i-1]) < self.epsilon:
                print(f"The model has found the optimal coefficients. \n iterations: {i} \n cost: {total_cost}")
                break

            if total_cost == np.inf:
                print('The model overshooted!')
                break

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias
    
    # Rescaling the weights and bias
    def coef_rescale(self):
        if self.scaled == True:
            scale = self.min_max_scale[1] - self.min_max_scale[0]

            self.weights = self.weights / scale
            self.bias = self.bias - np.sum(self.weights * self.min_max_scale[0])

            self.scaled = False