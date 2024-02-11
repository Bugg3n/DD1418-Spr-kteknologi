from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

#cd C:\Skrivbord\python\NER
# all datapoints
#python NER.py -d data/ner_training.csv -t data/ner_test.csv -b 
# 
# stochastic     
#python NER.py -d data/ner_training.csv -t data/ner_test.csv -s

# minibatch
#python NER.py -d data/ner_training.csv -t data/ner_test.csv -mgd


class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.1  # The learning rate.
    CONVERGENCE_MARGIN = 0.001#0.001  # The convergence criterion.
    MAX_ITERATIONS = 300 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1000 # Minibatch size (only for minibatch gradient descent)
    PLOT_FREQUENCY = 5
    LAMBDA = 0#.005#0.004
    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)
            # self.theta = [-2.5, 3.25, -1]

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)



    # ----------------------------------------------------------------------


    def loss(self, x, y):
        """
        Computes the loss function given the input features x and labels y
        
        :param      x:    The input features
        :param      y:    The correct labels
        """
        # REPLACE THE COMMAND BELOW WITH YOUR CODE
        
        result = 0
        
        # for a in range(len(x)):
        #     var = self.sigmoid(sum(np.multiply(self.theta,x[a])))
        #     if y[a] == 1:
        #         result += var
        #     if y[a] == 0:
        #         result += 1 - var

        for a in range(self.DATAPOINTS):
            var = self.sigmoid(sum(np.multiply(self.theta,x[a])))
            if y[a] == 1:
                result += math.log(var)
            if y[a] == 0:
                result += math.log(1 - var)
        
        result = -1 * result/len(y)
      
        return result


    def sigmoid(self, z):
        """
        The logistic function.
        """
        
        return 1.0 / ( 1 + np.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """
        
        # REPLACE THE COMMAND BELOW WITH YOUR CODE
        
        #conditional probability is given by theta
       
        var = np.multiply(self.theta , self.x[datapoint])
        
        prob = self.sigmoid(sum(var))
        
        
        if label == 1:
            return prob
        if label == 0:
            return 1 - prob


    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """
        
        # YOUR CODE HERE
        # 8e: slide 43
        for a in range(len(self.theta)): #For each 0 we calculate the gradient
        
            expr2 = []
            expr1 = np.multiply(self.theta,self.x)   ###(theta^T * x )       
            expr2 = self.sigmoid(expr1.sum(axis=1)) - 1 * self.y  #sigmoid(theta^T * x ) - Y
           
            delta_theta_a = np.multiply( np.array(self.x)[:,a] , expr2 ) # X * (sigmoid(theta^T * x ) - Y)
           
            self.gradient[a] = sum(delta_theta_a)/self.DATAPOINTS + self.LAMBDA * 2*self.theta[a] #!!!!!!!!!!!!!!!!!!!!!!!!!!
       
       

    def compute_gradient_minibatch(self, minibatch): #OBS la till y!!
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """
        # YOUR CODE HERE
        
        expr2 = []
        for a in range(len(self.theta)):
            expr1 = np.multiply(self.theta,self.x[minibatch[0]:minibatch[1]])   ###(theta^T * x )
            expr2 = self.sigmoid(expr1.sum(axis=1)) - 1 * self.y[minibatch[0]:minibatch[1]]

            delta_theta_a = np.multiply( np.array(self.x)[minibatch[0]:minibatch[1],a] , expr2 )
            self.gradient[a] = sum(delta_theta_a)/(minibatch[1] - minibatch[0])
        
        
        
        
        

    def compute_gradient(self, datapoint): #OBS la till y!!
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """
        
        # YOUR CODE HERE
        i = self.x[datapoint] 
        for a in range(len(self.theta)):
            expr1 = np.multiply(self.theta,i)   ###(theta^T * x )
            expr2 = self.sigmoid(sum(expr1)) + -1 * self.y[datapoint]   
            
            self.gradient[a] = i[a] * expr2 
        
            
    #python NER.py -d data/ner_training.csv -t data/ner_test.csv -s

    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        
        self.init_plot(self.FEATURES)
        
        # YOUR CODE HERE
        
        delta, iterations = 1 , 0
        while delta >= self.CONVERGENCE_MARGIN and iterations <= self.MAX_ITERATIONS:
            i = random.randint(0,self.DATAPOINTS)
            delta_L = []

            self.compute_gradient(i)
            
            self.theta = np.add(self.theta , -1 * self.LEARNING_RATE * self.gradient)
            iterations += 1
            for element in self.gradient:
                delta_L.append(pow(element,2))
            delta = sum(delta_L)   

            if iterations % self.PLOT_FREQUENCY == 0:
                self.update_plot(self.loss(self.x, self.y))

            iterations += 1
        
        
        


    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        
        
        self.init_plot(self.FEATURES)

        # YOUR CODE HERE
    
        k = 1
        delta, iterations = 1 , 0
        while delta >= self.CONVERGENCE_MARGIN and iterations <= self.MAX_ITERATIONS:
            delta_L = []
            if k * self.MINIBATCH_SIZE > len(self.x):
                i = [(k-1)*self.MINIBATCH_SIZE, len(self.x) - 1]
                k = 0
                
            else:
                i = [(k-1)*self.MINIBATCH_SIZE, k * self.MINIBATCH_SIZE]
            

            self.compute_gradient_minibatch(i)
            
            self.theta = np.add(self.theta , -1 * self.LEARNING_RATE * self.gradient)
            iterations += 1
            for element in self.gradient:
                delta_L.append(pow(element,2))
            delta = sum(delta_L)   
            

            # if iterations % self.PLOT_FREQUENCY == 0:
            #     self.update_plot(self.loss(self.x, self.y))
            k += 1
            
            iterations += 1   #obs denna konvergerar långsammare, öka max iterations

     

    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)

        
        # YOUR CODE HERE
        delta, iterations = 1 , 0
        while delta >= self.CONVERGENCE_MARGIN and iterations <= self.MAX_ITERATIONS:
            delta_L = []

            self.compute_gradient_for_all()
            



            self.theta = np.add(self.theta , -1 * self.LEARNING_RATE * self.gradient,self.LAMBDA*np.power(self.theta,2))
           
            for element in self.gradient:
                delta_L.append(pow(element,2))
            delta = sum(delta_L)
            
            if iterations % self.PLOT_FREQUENCY == 0:
                print(iterations)
                self.update_plot(self.loss(self.x, self.y))
           
            # print(100 *iterations/self.MAX_ITERATIONS, "%")
            iterations +=1
        
           
            
        
        


    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:');

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]), "theta")
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]), "gradient")


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, 1+max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()


if __name__ == '__main__':
    main()

#Med regularisering 0.04
# Model training took 118.6s
# Model parameters:
# 0: -2.0521  1: 2.0755  2: -0.1050
#                        Real class
#                         0        1
# Predicted class:  0 83831.000 3242.000
#                   1  879.000 12046.000
#


# Med regularisering 0.035
# Model training took 246.47s
# Model parameters:
# 0: -2.1626  1: 2.2341  2: 0.1406
#                        Real class
#                         0        1
# Predicted class:  0 80674.000 2015.000
#                   1 4036.000 13273.000


# utan regularisering
# Model parameters:
# 0: -2.5317  1: 3.0392  2: -0.3306
#                        Real class
#                         0        1
# Predicted class:  0 80674.000 2015.000
#                   1 4036.000 13273.000
# Press Return to finish the program...
