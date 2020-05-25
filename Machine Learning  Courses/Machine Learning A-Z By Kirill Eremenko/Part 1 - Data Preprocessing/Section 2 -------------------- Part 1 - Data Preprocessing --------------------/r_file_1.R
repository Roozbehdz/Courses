#Data Preprocessing

#Importing the dataset
dataset = read.csv('Data.csv')

library(caTools)
set.seed(123)

split = sample.split(dataset$)