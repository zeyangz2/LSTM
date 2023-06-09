# LSTM
train LSTM with stroke therapist and patient data

We collected some sample data in file trial2.csv to train the network.

Feel free to change the hyperparameters in config_1.json. Recommand to directly use config_2.json.

Uncomment the lines in run.py to save your predicted results.

Use this jupyter notebook to get a 3D graph of predicted trajectory. You can use the results I got from file: saved test&predict results
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeyangz2/LSTM/blob/master/3D%20graph%20LSTM%20results.ipynb)

Since LSTM is good for point by point prediction but not good for long sequence prediction, we use another model called Informer to do long sequence prediction.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeyangz2/LSTM/blob/master/Informer_for_stroke_research.ipynb)

To use Informer, you must need to add a date column and put the feaeture that you want to predict to the rightmost column in your data file(.scv file). Right now, Informer gives proper predicted results but still not good enough.

Make sure to edit the sample data in trial2.csv before using Informer to train the network.

Architecture of LSTM model:

![model](https://github.com/zeyangz2/LSTM/assets/73300066/801da6f7-11a1-4b94-bb89-f78f34086115)

results:

red line for ground truth and green line for our prediction

![3D result](https://github.com/zeyangz2/LSTM/assets/73300066/567839c0-16eb-4870-8e89-90343fa5423b)

z_axis:

![result_z](https://github.com/zeyangz2/LSTM/assets/73300066/e4a386d9-3d7d-46bc-b2ee-57cbad2cc750)

y_axis:

![results_y](https://github.com/zeyangz2/LSTM/assets/73300066/8902b71a-0f5a-4329-9964-554f8a639a62)

x_axis:

![results_x](https://github.com/zeyangz2/LSTM/assets/73300066/7b59c058-e4d3-4e12-a590-9ef38c316835)

more updates later...


