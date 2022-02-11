### About The Project

<p align="center">
   <img src="https://github.com/usama-akram-gt/LUNAR-LANDAR-SIMULATOR/blob/master/assets/game_play.gif" />
</p>

Deep learning to help land lander onto his target. Collected the data by playing the game and then trained the network
from scratch using single layer neural network where i struggled mostly towards pre-processing data which makes it
difficult to land. Our network receives two inputs and then accordingly predict the right move for the lander. Trained
the network and saved the weights over the file for later use during prediction and uses RMSE for the performance 
measure.

### Pre-Processing
* There was alot of redundancy over the data like NA values resolved by removing entire rows <br>
* Reduces the data dimentionality upto 3 decimal points <br>
* Removed the duplicate rows and lower the size of the data with upto 80% winning moves while 30% remaining

<p align="center">
    <img src="https://github.com/usama-akram-gt/LUNAR-LANDAR-SIMULATOR/blob/master/assets/preprocessing2.jpg" width="100" />
    <img src="https://github.com/usama-akram-gt/LUNAR-LANDAR-SIMULATOR/blob/master/assets/preprocessing2.jpg" width="100" /> 
    <img src="https://github.com/usama-akram-gt/LUNAR-LANDAR-SIMULATOR/blob/master/assets/preprocessing2.jpg" width="100" />
</p>

## Important information related to files
* Weights.txt file has the weights getting saved on each run
* Weights_for_prediction.txt has the weights game will use (don't forget to copy weights from weights.txt to here after the training done)

### Built With
* [Python](https://www.python.org) <br>
For downloading python visit: <a href="https://www.python.org/downloads/">Click Here</a>

### Installation
1. Clone the repo
   ```sh
   git clone https://github.com/usama-akram-gt/LUNAR-LANDAR-SIMULATOR.git
   ```
2. Install python-pip packages @root
   ```sh
   pip install -r requirements.txt
   ```
3. Run this for training
   ```sh
   python SingleNeuralNetwork.py
   ```
4. Run this for gameplay
   ```sh
   python Main.py
   ```
   
### Results
* Game did perform well when used LR = 0.1, Momentum = 0.01, Lambda = 0.1 <br>
<p align="center">
    <img src="https://github.com/usama-akram-gt/LUNAR-LANDAR-SIMULATOR/blob/master/assets/hyperparameters.jpg" />
</p>


## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
