import math
import random
from csv import reader

our_lambda = 0.1 # here out 0.1 is lambda
learning_rate = 0.1 # defining learning rate for back propagation
momentum_mt = 0.1 # our momentum same as like lambda kind of

stopping_error_check = 0.0 # storing errors into it so that we can check upto 5 decimals no change for 2 continues iterations

# A neural network class
class Neuron:  
      
    # init method or constructor   
    def __init__(self, AV, number_of_weights, index):  
        self.AV = AV # setting neuron Activation Value
        # weights list
        
        # defining 2 more variables for the back propogation
        self.delta_weights = []
        self.grad_val = 0

        self.weights_list = [] # randomizing weights of the neuron according to the no. of weights
        for i in range(0, number_of_weights):
            self.weights_list.append(random.random())
            self.delta_weights.append(0)

        self.index = index # setting neuron index

    def sigmoid(self, x):
        self.AV = 1 / (1 + math.exp(-(our_lambda * x))) #applying the sigmoid function

    def mult_weights(self, prevLayer):
        total_sum = 0
        for i in range(0, len(prevLayer)): # calculating weights by multiplying AV of each neuron with weights and adding
            total_sum = total_sum + (float(prevLayer[i].AV) * float(prevLayer[i].weights_list[self.index]))
        
        self.sigmoid(total_sum) 


#2*2*2 structure

#input layer (where third object is the bias neuron)
input_layer = [Neuron(0, 2, 0), Neuron(0, 2, 1), Neuron(0, 2, 2)]
#hidden layer (where third object is the bias neuron)
hidden_layer = [Neuron(0, 2, 0), Neuron(0, 2, 1), Neuron(0, 2, 2)]
#output layer
output_layer = [Neuron(0, 0, 0), Neuron(0, 0, 1)]

def feed_forward_process(inputs):
    # Setting input_layer activation values
    for i in range(0, len(input_layer)):
        input_layer[i].AV = inputs[i]
    
    # Calculating hidden_layer activation values on the basis of input_layer
    for i in range(0, (len(hidden_layer) - 1)):
        hidden_layer[i].mult_weights(input_layer)

    hidden_layer[-1].AV = 1 #It's our bios which is for 3rd neuron we are setting up on our hidden layer which value always be 1

    # Calculating output_layer activation values on the basis of hidden_layer
    for i in range(0, len(output_layer)):
        output_layer[i].mult_weights(hidden_layer)

    return

# we sent our actual outputs that we expect our feed forward network to predict
def back_propogation_process(outputs):
    error = []

    # first calculating errors where we subtract the expected output which we pass as parameter to the 
    # back propogation and then subtract it with the actual result we get for our output which is stored in Activation value of our layer
    for i in range(0, len(output_layer)):
        error.append(float(outputs[i]) - float(output_layer[i].AV))

    # now calculating gradiant decent for output layer which uses error which we calculated 
    for i in range(0, len(output_layer)):
        output_layer[i].grad_val = our_lambda * output_layer[i].AV * (1 - output_layer[i].AV) * error[i] 

    # now calculating gradiant decent for out hidden layer where we don't use any error
    for i in range(0, len(hidden_layer)):
        # now in this loop we first multiply the calculated gradiant decent with the weights of the hidden layer first for which we needed the addition result which we add
        result = 0
        for j in range(0, len(output_layer)):
            result = result + ( float(output_layer[j].grad_val) * float(hidden_layer[i].weights_list[j]) ) 
        
        hidden_layer[i].grad_val = our_lambda * hidden_layer[i].AV * (1 - hidden_layer[i].AV) * result

    
    # for now as we have calculated gradiatn decent for both the hidden and output layer so now it's time to calculate weight updations for neurons
    for i in range(0, len(hidden_layer)):
        for j in range(0, len(output_layer)): # We have momentum same as like lambda where we tell it how fast it should move?
            hidden_layer[i].delta_weights[j] = learning_rate * float(output_layer[j].grad_val) * float(hidden_layer[i].AV) + momentum_mt * float(hidden_layer[i].delta_weights[j])
    
    #  time for us to calculate updated weights for the input layer
    for i in range(0, len(input_layer)):
        for j in range(0, (len(hidden_layer) - 1)): # We have momentum same as like lambda where we tell it how fast it should move?
            input_layer[i].delta_weights[j] = learning_rate * float(hidden_layer[j].grad_val) * float(input_layer[i].AV) + momentum_mt * float(input_layer[i].delta_weights[j])
    
    # now finally updating weights after calculating new delta weights on the basis of gradiant
    for i in range(0, len(hidden_layer)):
        for j in range(0, len(hidden_layer[i].weights_list)): # We have momentum same as like lambda where we tell it how fast it should move?
            hidden_layer[i].weights_list[j] = float(hidden_layer[i].weights_list[j]) + float(hidden_layer[i].delta_weights[j])

    for i in range(0, len(input_layer)):
        for j in range(0, len(input_layer[i].weights_list)): # We have momentum same as like lambda where we tell it how fast it should move?
            input_layer[i].weights_list[j] = float(input_layer[i].weights_list[j]) + float(input_layer[i].delta_weights[j])

    # Note: we don't calculate any gradiant decent for our input layer

    return
    

def training():
    # start reading here and keep passing that to our feed_forward_process and 
    # that will end as file stops reading from the data
    with open('game_training.csv', 'r') as training:
        training_csv = reader(training)
        for row in training_csv:
            feed_forward_process([row[0], row[1], 1])
            back_propogation_process([row[2], row[3]])


# Feedforward + error calculation // but on the training file
def training_error(): # We will be calculating this error on our training data after we are done with the training
    total_error = []
    with open('game_training.csv', 'r') as training:
        training_csv = reader(training)
        for row in training_csv:
            feed_forward_process([row[0], row[1], 1]) #Here row
            # first we will kept calculating error for each row and then keep appending it and then 
            total_error.append( ( (float(row[2]) - float(output_layer[0].AV))**2 + (float(row[3]) - float(output_layer[1].AV))**2 ) / 2)               
        
        # calculating our error here in training error and checking it if the error 
        # become stable then we stop but I need to ask teacher how we check actually 
        # check that now we are not getting any significant change in our model
        print('Training Error: ')
        return root_mean_squere(total_error)

# Feedforward + error calculation // but on the validation/test file
def validation_error(): # We will be doing it on our testing data or we can call it validation data
    total_error = []
    with open('game_validation.csv', 'r') as training:
        training_csv = reader(training)
        for row in training_csv:
            feed_forward_process([row[0], row[1], 1]) #Here row
            # first we will kept calculating error for each row and then keep appending it and then
            total_error.append( ( (float(row[2]) - float(output_layer[0].AV))**2 + (float(row[3]) - float(output_layer[1].AV))**2 ) / 2)                
                            
        print('Validation Error: ')
        root_mean_squere(total_error) 
    
'''
RMSE of test > RMSE of train => OVER FITTING of the data
RMSE of test < RMSE of train => UNDER FITTING of the data
'''
def root_mean_squere(total_error):
    # We will be kept appending all our errors which we will be calculating
    # and then we will devide it by the total number or error lines which have got
    # calculated and then taking squere of it and that will be our error.   
    print(math.sqrt(sum(total_error) / len(total_error)))
    return math.sqrt(sum(total_error) / len(total_error))

def saving_weights():
    weights_file = open("weights.txt", "a")
    weights_file.write("\n")
    # Iterating over input_layer and weights after epoch for saving them to file
    for i in range(0, len(input_layer)):
        for j in range(0, len(input_layer[i].weights_list)):
            weights_file.write(str(input_layer[i].weights_list[j]) + ",")

    # Iterating over hidden_layer and weights after epoch for saving them to file
    for i in range(0, len(hidden_layer)):
        for j in range(0, len(hidden_layer[i].weights_list)):
            weights_file.write(str(hidden_layer[i].weights_list[j]) + ",")        
    weights_file.close()
    return

def load_weights():
    # Picking them from the file
    weights = []
    with open('weights_for_prediction.txt') as f:
        line = f.read()
        weights = line.split(",")        

    k = 0
    # Iterating over input_layer and weights for assigning
    for i in range(0, len(input_layer)):
        for j in range(0, len(input_layer[i].weights_list)):
            input_layer[i].weights_list[j] = weights[k]
            k = k + 1
    
    # Iterating over hidden_layer and weights for assigning
    for i in range(0, len(hidden_layer)):
        for j in range(0, len(hidden_layer[i].weights_list)):
            hidden_layer[i].weights_list[j] = weights[k]
            k = k + 1
    return

# specifying min and maximum values according to model
x1_max = 477.731
x2_max = 567.535
y1_max = 7.995
y2_max = 4.339

x1_min = -542.953
x2_min = 65.460
y1_min = -3.866
y2_min = -5.147

def normalization(inputs):
    input1 = (inputs[0] - (x1_min)) / ((x1_max) - (x1_min))  # Normalizing first data point from input row
    input2 = (inputs[1] - (x2_min)) / ((x2_max) - (x2_min))  # Normalizing second data point from input row
    return [input1, input2]

def de_normalization(outputs):
    # As we have normalized now need to convert it back to the original state using de_normalization
    return [(outputs[0] * ((y1_max) - (y1_min)) + (y1_min)),(outputs[1] * ((y2_max) - (y2_min)) + (y2_min))]

def prediction(input_row):
    # we predict in this function that what would be the result against one game input
    normalized_result = normalization(input_row) # Normalizing the input row before passing to feed_forward
    normalized_result.append(1) # Appending bias into normalized list of inputs
    print(normalized_result)
    
    load_weights() # Loading weights
    
    feed_forward_process(normalized_result) # Checking feed_forward_ with the input row
    
    print([output_layer[0].AV,output_layer[1].AV])
    return de_normalization([output_layer[0].AV,output_layer[1].AV]) # returning denormalized predicted result to the game 

run = 1
if run == 1:
    for epoch_count in range(2000): # Controlling no. of epochs in case if the model will not in any case
        print('Epoch no. ', epoch_count + 1)
        
        # We call this function to train our model using the game dataset
        training()  
        
        # Checking training error after we are done training our model
        error = training_error()

        # Checking if upto 5 decimals we don't see any siginifant change then stopping
        if float("{:.5f}".format(error)) == float("{:.5f}".format(stopping_error_check)):
            print("Stopping")
            break
        else:
            stopping_error_check = error

        # Checking validation error on our validation data after training
        validation_error() 
        print('\n')

        # Weights are saved after every epoch
        saving_weights()

        #increasing epoch size
        epoch_count += 1

elif run == 2:
    print(prediction([273.670,354.100])) # Making prediction