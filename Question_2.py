import pandas as pd
import matplotlib.pyplot as plt

#Load the crime dataset
crime_Dataset = pd.read_csv("crime1.csv")
#Select the ViolentCrimesPerPop column in the dataset
value_of_violent_Crimes = crime_Dataset["ViolentCrimesPerPop"]

# Histogram of ViolentCrimesPerPop
plt.hist(value_of_violent_Crimes,bins=20)
plt.title("Histogram of Violent Crimes Per Population")
plt.xlabel("Violent_Crimes_Per_Population")
plt.ylabel("Frequency")
plt.show()


# BoxPlot of ViolentCrimesPerPop
plt.boxplot(value_of_violent_Crimes)
plt.title("Box Plot of Violent Crimes Per Population")
plt.xlabel("Violent_Crimes_Per_Population")
plt.ylabel("Value")
plt.show()

# This histogram shows that the data is right-skewed. This is seen as most of the data points are concentrated on the lower end of the scale, mostly between 0.1 and 0.5, and the data seems to taper off when approaching the higher values.
# The median is shown to be at approximately 0.38. This median indicates that 50% of the recorded areas have a violent crime value below 0.38 and 50% are above. As the median is located closer to the bottom of the box, it reinforces the previous observation of the data being right-skewed.
# This box plot does not suggest the presence of outlier. On a boxplot, an outlier would be indicated by individual dots located outside of the whiskers. There are no dots on this boxplot and the whiskers seem to extend all the way to the minimum and maximum values.