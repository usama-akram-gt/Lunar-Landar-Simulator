# Min Max Normalization
'''
y = (x – min) / (max – min)
'''
# Numbers of Column
'''
We have got 4 column after playing game named as (x1, x2, y1, y2)
'''

# Number of Rows
'''
We have got 12,380 total rows after playing game.
'''

# Maximum and Minimum from each column
'''
We have got 4 columns after playing game
# x1, x2, y1, y2
# Maximum from each column:
# 766.799120 -> x1
# 1006.451787 -> x2
# 8.000000 -> y1
# 7.765852 -> y2
# Minimum from each column:
# -800.389713 -> x1
# 65.113279 -> x2
# -5.920036 -> y1
# -7.509682 -> y2
'''

import pandas as pd
from sklearn import preprocessing

game_data_frame = pd.read_csv("./ce889_dataCollection.csv")





x = game_data_frame.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
game_data_frame = pd.DataFrame(x_scaled)

game_data_frame.to_csv('final.csv', index=False)  