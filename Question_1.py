import pandas as pd
import numpy as np

#Load the crime dataset
crime_Dataset = pd.read_csv("crime1.csv")
#Select the ViolentCrimesPerPop column in the dataset
value_of_violent_Crimes = crime_Dataset["ViolentCrimesPerPop"]

#Compute mean, median, standard deviation, minimum, and maximum values
mean_value = value_of_violent_Crimes.mean()
median_value = value_of_violent_Crimes.median()
standard_deviation_value = value_of_violent_Crimes.std()
minimum_value = value_of_violent_Crimes.min()
maximum_value = value_of_violent_Crimes.max()

#Print the sesults
print(f"Mean:", mean_value)
print(f"Median:", median_value)
print(f"Standard Deviation:", standard_deviation_value)
print(f"Minimum:", minimum_value)
print(f"Maximum:", maximum_value)


# When comparing the mean and the median, it is observed that the mean (0.441) is greater than the median (0.39). This indicates that the distribution of Violent Crimes per Population is right-skewed. This means that there are certain communities with very high violent crime rates which lead to the mean to increase

# The maximum value is 1.0 and the minimum value is 0.02, and these extreme values affect the mean more than they affect the median. This is because the mean uses every single data point in its calculation, whereas the median only depends on the middle value of the data and is therefore much more resistant to extreme outliers