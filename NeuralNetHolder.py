from SingleNeuralNetwork import prediction
class NeuralNetHolder:

    def __init__(self):
        super().__init__()

    
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        input_row_list = input_row.split(",")
        output_row_list = prediction([float(input_row_list[0]), float(input_row_list[1])])
        return output_row_list
