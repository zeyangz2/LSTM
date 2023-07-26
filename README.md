# LSTM for therapist trajectory prediction
train LSTM with stroke therapist and patient data

We collected some sample data in data file to train the network.

Feel free to change the hyperparameters in config_1.json. Recommand to directly use config_2.json, since the hyperparameters are fine-tuned by Zeyang.

Uncomment the lines in run.py to save your predicted results.

Use this jupyter notebook to get a 3D graph of predicted trajectory. You can use the results I got from file: saved test&predict results
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeyangz2/LSTM/blob/master/3D%20graph%20LSTM%20results.ipynb)

Since LSTM is good for point by point prediction but not good for long sequence prediction, we use another model called Informer to do long sequence prediction.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeyangz2/LSTM/blob/master/Informer_for_stroke_research.ipynb)

To use Informer, you must need to add a date column and put the feaeture that you want to predict to the rightmost column in your data file(.scv file). Right now, Informer gives proper predicted results but still not good enough.

Make sure to edit the sample data in trial2.csv before using Informer to train the network.

## Architecture of new LSTM model 

(the number of inputs given to the model could change depending on the number of inputs we want to use to train the model):

![4360afb28290abd2a1b6080a8f3a115](https://github.com/zeyangz2/LSTM/assets/73300066/8af77e54-3ac4-43b2-abb7-656a0de2b296)

# results from trial2.csv:

red line for ground truth and green line for our prediction

![3D result](https://github.com/zeyangz2/LSTM/assets/73300066/567839c0-16eb-4870-8e89-90343fa5423b)

## **z_axis:**

![result_z](https://github.com/zeyangz2/LSTM/assets/73300066/e4a386d9-3d7d-46bc-b2ee-57cbad2cc750)

**z_axis model loss:**

![Model Loss z](https://github.com/zeyangz2/LSTM/assets/73300066/cf8b165b-11d7-4b07-bf97-91d8ae930024)


## **y_axis:**

![results_y](https://github.com/zeyangz2/LSTM/assets/73300066/8902b71a-0f5a-4329-9964-554f8a639a62)

**y_axis model loss:**

![Model Loss y](https://github.com/zeyangz2/LSTM/assets/73300066/00481fb4-db25-4b21-a9e7-a17d4dc4da6a)


## **x_axis:**

![results_x](https://github.com/zeyangz2/LSTM/assets/73300066/7b59c058-e4d3-4e12-a590-9ef38c316835)

**x_axis model loss:**

![Model Loss x](https://github.com/zeyangz2/LSTM/assets/73300066/ebb09931-97cb-4294-9342-2363734c9429)

# results from srikar and zeyang_1.csv:

red line for ground truth and green line for our prediction

![37598975822b4d1736266b1c8c183d3](https://github.com/zeyangz2/LSTM/assets/73300066/7ae01ef7-6ad4-4208-b2d1-8721e824b0e4)


## **z_axis:**

![new result z](https://github.com/zeyangz2/LSTM/assets/73300066/4e492782-00c2-4d93-bfaa-0b2202bd572b)


**z_axis model loss:**


![new Model Loss z](https://github.com/zeyangz2/LSTM/assets/73300066/e3aa0d03-8d37-46f5-8b52-da24600cb54a)


## **y_axis:**

![new result y](https://github.com/zeyangz2/LSTM/assets/73300066/910daef8-f1c1-499c-8147-9d51ba8538bd)


**y_axis model loss:**

![new Model Loss y](https://github.com/zeyangz2/LSTM/assets/73300066/02ca9279-3683-446d-9068-52a9afe0236f)



## **x_axis:**

![new result x](https://github.com/zeyangz2/LSTM/assets/73300066/cfc7cfe6-4aaa-4e85-b776-69a0d8b951c4)


**x_axis model loss:**

![new Model Loss x](https://github.com/zeyangz2/LSTM/assets/73300066/a5a75cf3-9232-4fc5-bf0b-f632938b5b77)



more updates later...


