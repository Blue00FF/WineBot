# WineBot

In this project, the aim is to create a program in Python that uses a Supporting Vector Machine to train on wine descriptions from a kaggle wine reviews database, 
then takes as first input from the user information about the colour, price, point range and country of origin of the desired wine and as second input any desired 
flavour notes. The information from the first input is then used to filter the database entries, while the second one is used to give each remaining entry a score 
that indicates how close it is to the key words indicated in the flavour notes (sweetness, sourness, fruity, etc.). The wine with the highest score and its 
description are then printed to console and the program stops. In case the filtering returned an empty databse, i.e. no such wine exists in the database adhering to 
such specifications, the program prints to console that it couldn't find any wine to specifications.

In order to run the program for yourself, you must first download the database from here (), extract it and copy the csv files to a folder named "data" in the same folder as the "main" one and not inside the "main" folder. Run the "data_skimmer.py" first to drop the unneeded columns and then run the "main.py" if you wish to insert only two inputs and for the program to look for key words in your answer or the "simple.py" if you wish for the program to ask for separate inputs (colour first, then price, then point range and so on). Both work exactly the same.   
