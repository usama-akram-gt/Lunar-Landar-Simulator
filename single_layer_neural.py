import math
import random
from csv import reader

# A neural network class
class Neuron:  
      
    # init method or constructor   
    def __init__(self, AV, number_of_weights, index):  
        self.AV = AV
        # weights list
        
        self.weights_list = []
        for i in range(0, number_of_weights):
            #generates random values between 0 to 1
            self.weights_list.append(random.random())

        #print(self.weights_list)

        self.index = index

        #now defining 2 more variables for the back propogation
        self.delta_weights = [0, 0]
        self.grad_val = 0

    def sigmoid(self, x):
        #applying the sigmoid function
        self.AV = 1 / (1 + math.exp(-(0.3 * x)))
        #print(self.AV)  

    def mult_weights(self, prevLayer):
        total_sum = 0
        for i in range(0, len(prevLayer)):
            total_sum = total_sum + (float(prevLayer[i].AV) * float(prevLayer[i].weights_list[self.index]))
        
        self.sigmoid(total_sum) 


if __name__ == "__main__":

    #predicting XOR table through this single layer neural network
    '''
    v1  v2 | output
    0   0  |   0
    0   1  |   1
    1   0  |   1
    1   1  |   0
    '''

    #input layer (where third object is the bias neuron)
    input_layer = [Neuron(0, 2, 0), Neuron(0,2,1), Neuron(0, 2, 2)]
    #hidden layer (where third object is the bias neuron)
    hidden_layer = [Neuron(0, 2, 0), Neuron(0, 2, 1), Neuron(0, 2, 2)]
    #output layer
    output_layer = [Neuron(0, 0, 0), Neuron(0, 0, 1)]
    
    #third value is the bias value
    #inputs = [1, 1, 1] instead of like this we will be reading them from the list with output as well
    # We will be reading data from text file


    def feed_forward_process(inputs):

        for i in range(0, len(input_layer)):
            input_layer[i].AV = inputs[i]
        
        #hard coding weights
        '''
        input_layer[0].weights_list[0] = -5.1
        input_layer[0].weights_list[1] = -4.3
        input_layer[1].weights_list[0] = 5.2
        input_layer[1].weights_list[1] = 4.0
        input_layer[2].weights_list[0] = 2.7
        input_layer[2].weights_list[1] = -2.2
        input_layer[0].weights_list[0] = 36.1536705576
        input_layer[0].weights_list[1] = 0.584834730258
        input_layer[1].weights_list[0] = 1.17265681602
        input_layer[1].weights_list[1] = -18.3618481024
        input_layer[2].weights_list[0] = -23.0049550389
        input_layer[2].weights_list[1] = 15.8550490139
        '''

        hidden_layer[0].mult_weights(input_layer)
        hidden_layer[1].mult_weights(input_layer)

        hidden_layer[2].AV = 1 #It's our bios which is for 3rd neuron we are setting up on our hidden layer which value always be 1
        '''
        hidden_layer[0].weights_list[0] = -6.0
        hidden_layer[1].weights_list[0] = 6.5
        hidden_layer[2].weights_list[0] = 2.7
        hidden_layer[0].weights_list[0] = 1.44136083822
        hidden_layer[0].weights_list[1] = -15.3524786656
        hidden_layer[1].weights_list[0] = 1.02460673772
        hidden_layer[1].weights_list[1] = 41.8405191093
        hidden_layer[2].weights_list[0] = -0.989759205097
        hidden_layer[2].weights_list[1] = -1.54938717042
        '''

        '''
        for i in range(0, len(output_layer)):
            output_layer[i].mult_weights(hidden_layer)
        '''
        output_layer[0].mult_weights(hidden_layer)
        output_layer[1].mult_weights(hidden_layer)

        #print('Our prediction result is: ', output_layer[0].AV)

    # we sent our actual outputs that we expect our feed forward network to predict
    def back_propogation_process(outputs):
        error = []
        our_lambda = 0.3 # here out 0.1 is lambda
        momentum_mt = 0.2 # our momentum same as like lambda kind of

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
        #for i in range(0, len(hidden_layer)):
            #for j in range(0, len(output_layer)): # We have momentum same as like lambda where we tell it how fast it should move?
                #hidden_layer[i].delta_weights[j] = our_lambda * float(output_layer[j].grad_val) * float(hidden_layer[i].AV) + momentum_mt * float(hidden_layer[i].delta_weights[j])
        hidden_layer[0].delta_weights[0] = our_lambda * float(output_layer[0].grad_val) * float(hidden_layer[0].AV) + momentum_mt * float(hidden_layer[0].delta_weights[0])
        hidden_layer[0].delta_weights[1] = our_lambda * float(output_layer[1].grad_val) * float(hidden_layer[0].AV) + momentum_mt * float(hidden_layer[0].delta_weights[1])
        hidden_layer[1].delta_weights[0] = our_lambda * float(output_layer[0].grad_val) * float(hidden_layer[1].AV) + momentum_mt * float(hidden_layer[1].delta_weights[0])
        hidden_layer[1].delta_weights[1] = our_lambda * float(output_layer[1].grad_val) * float(hidden_layer[1].AV) + momentum_mt * float(hidden_layer[1].delta_weights[1])
        #  time for us to calculate updated weights for the input layer

        #for i in range(0, len(input_layer)):
            #for j in range(0, (len(hidden_layer) - 1)): # We have momentum same as like lambda where we tell it how fast it should move?
                #input_layer[i].delta_weights[j] = our_lambda * float(hidden_layer[j].grad_val) * float(input_layer[i].AV) + momentum_mt * float(input_layer[i].delta_weights[j])
        input_layer[0].delta_weights[0] = our_lambda * float(hidden_layer[0].grad_val) * float(input_layer[0].AV) + momentum_mt * float(input_layer[0].delta_weights[0])
        input_layer[0].delta_weights[1] = our_lambda * float(hidden_layer[1].grad_val) * float(input_layer[0].AV) + momentum_mt * float(input_layer[0].delta_weights[1])
        input_layer[1].delta_weights[0] = our_lambda * float(hidden_layer[0].grad_val) * float(input_layer[1].AV) + momentum_mt * float(input_layer[1].delta_weights[0])
        input_layer[1].delta_weights[1] = our_lambda * float(hidden_layer[1].grad_val) * float(input_layer[1].AV) + momentum_mt * float(input_layer[1].delta_weights[1])

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
        with open('game_final_accurate.csv', 'r') as training:
            training_csv = reader(training)
            for row in training_csv:
                feed_forward_process([row[0], row[1], 1])
                back_propogation_process([row[2], row[3]])


    # Feedforward + error calculation // but on the training file
    def training_error(): # We will be calculating this error on our training data after we are done with the training
        total_error = []
        with open('game_final_accurate.csv', 'r') as training:
            training_csv = reader(training)
            for row in training_csv:
                feed_forward_process([row[0], row[1], 1]) #Here row
                # first we will kept calculating error for each row and then keep appending it and then 
                total_error.append( ( (float(row[2]) - float(output_layer[0].AV))**2 + (float(row[3]) - float(output_layer[1].AV))**2 ) / 2)               
            
            # calculating our error here in training error and checking it if the error 
            # become stable then we stop but I need to ask teacher how we check actually 
            # check that now we are not getting any significant change in our model
            print('Training Error: ')
            root_mean_squere(total_error)         

    # Feedforward + error calculation // but on the validation/test file
    def validation_error(): # We will be doing it on our testing data or we can call it validation data
        total_error = []
        with open('game_validation_file.csv', 'r') as training:
            training_csv = reader(training)
            for row in training_csv:
                feed_forward_process([row[0], row[1], 1]) #Here row
                # first we will kept calculating error for each row and then keep appending it and then
                total_error.append( ( (float(row[2]) - float(output_layer[0].AV))**2 + (float(row[3]) - float(output_layer[1].AV))**2 ) / 2)                
                                
            # calculating our error here in training error and checking it if the error 
            # become stable then we stop but I need to ask teacher how we check actually 
            # check that now we are not getting any significant change in our model
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
        weights_file = open("new_game_data_weights_accurate.txt", "a")
        weights_file.write("\n")
        weights_file.write(
                str(input_layer[0].weights_list[0]) + "," +
                str(input_layer[0].weights_list[1]) + "," +  
                str(input_layer[1].weights_list[0]) + "," + 
                str(input_layer[1].weights_list[1]) + "," +
                str(input_layer[2].weights_list[0]) + "," + 
                str(input_layer[2].weights_list[1]) + "," + 
                str(hidden_layer[0].weights_list[0]) + "," +
                str(hidden_layer[0].weights_list[1]) + "," + 
                str(hidden_layer[1].weights_list[0]) + "," + 
                str(hidden_layer[1].weights_list[1]) + "," +
                str(hidden_layer[2].weights_list[0]) + "," + 
                str(hidden_layer[2].weights_list[1]))
        weights_file.close()
        return

    def load_weights():
        # Picking them from the file
        weights = []
        with open('/Users/mac/Downloads/Deep Learning & Neural Network Lab/Lab1/single_line_weight.txt') as f:
            line = f.read()
            weights = line.split(",")        

        input_layer[0].weights_list[0] = weights[0]
        input_layer[0].weights_list[1] = weights[1]
        input_layer[1].weights_list[0] = weights[2]
        input_layer[1].weights_list[1] = weights[3]
        input_layer[2].weights_list[0] = weights[4]
        input_layer[2].weights_list[1] = weights[5]
        hidden_layer[0].weights_list[0] = weights[6]
        hidden_layer[0].weights_list[1] = weights[7]
        hidden_layer[1].weights_list[0] = weights[8]
        hidden_layer[1].weights_list[1] = weights[9]
        hidden_layer[2].weights_list[0] = weights[10]
        hidden_layer[2].weights_list[1] = weights[11]
        return

    def normalization(outputs):
        output1 = (outputs[0] - (-599.480)) / ((750.731) - (-599.480))
        output2 = (outputs[1] - (65.754)) / ((719.332) - (65.754))
        return [output1, output2]

    def de_normalization(outputs):
        # As we have normalized now need to convert it back to de_normalize 
        return [(outputs[0] * ((7.999) - (-4.408)) + (-4.408)),(outputs[1] * ((6.193) - (-7.876)) + (-7.876))]

    def prediction():
        # we predict in this function that what would be the result against one game input
        normalized_result = normalization([273.670,354.100])
        normalized_result.append(1)
        print(normalized_result)
        load_weights()
        feed_forward_process(normalized_result) #Here row
        print([output_layer[0].AV,output_layer[1].AV])
        return de_normalization([output_layer[0].AV,output_layer[1].AV])

    for i in range(1):
        print(prediction())
        '''
        print('Epoch no. ', i + 1)
        # We call this function to train our model using the game dataset
        training()  
   
        # Checking training error after we are done training our model
        training_error() 
        # Checking validation error on our validation data after training
        #validation_error() 
        print('\n')

        # Weights are saved after every epoch
        saving_weights()
        '''