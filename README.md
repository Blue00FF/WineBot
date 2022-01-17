# WineBot

(Inspired by an idea by Ari Virrey)

In this project, the aim is to create a program in Python that uses Term Frequency-Inverse Document Frequency to train on wine descriptions from a kaggle wine reviews database, 
then takes as first input from the user information about the colour, price, star rating and country of origin of the desired wine and as second input any desired 
flavour notes. The information from the first input is then used to filter the database entries, while the second one is used to give each remaining entry a score 
that indicates how close it is to the key words indicated in the flavour notes (sweetness, sourness, fruity, etc.). The wine with the highest score and its 
description are then printed to console and the program stops. In case the filtering returned an empty databse, i.e. no such wine exists in the database adhering to 
such specifications, the program prints to console that it couldn't find any wine to specifications.

In order to run the program for yourself, you must first download the database from here (https://www.kaggle.com/zynicide/wine-reviews), extract it in the same folder as the files you downloaded from my repository. To install the required modules, you only need to open a terminal window in the folder (shift + right click in the folder, then click on "open in terminal"/"open powershell window here") and execute the command "pip install -r requirements.txt". To execute the program you open a terminal window and execute the command "python -m main" after having installed the requirements.

Note that you will need to have python >= 3.8 since I make use of the Walrus operator.