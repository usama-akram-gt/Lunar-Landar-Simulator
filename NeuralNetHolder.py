from SingleNeuralNetwork import prediction # importing our prediction function from our single_layer_neural_network
class NeuralNetHolder:

    def __init__(self):
        super().__init__()

    
    def predict(self, input_row):
        input_row_list = input_row.split(",") # Getting input_row from game as string and convering it to list for prediction 
        output_row_list = prediction([float(input_row_list[0]), float(input_row_list[1])]) # predicting it using our model
        return output_row_list # return ing it back to game so that game could make move accordingly.
