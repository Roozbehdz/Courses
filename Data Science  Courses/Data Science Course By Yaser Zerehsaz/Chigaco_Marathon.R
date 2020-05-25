rm(list=ls(all.names=TRUE))
Finish = read.csv('C:/Users/10/Documents/R Scripts/Zerehsaz Data Science Course/finish.csv')
Age = read.csv('C:/Users/10/Documents/R Scripts/Zerehsaz Data Science Course/Age.csv')

Finish=as.matrix(Finish)
Age=as.matrix(Age)

#Fit a linear model
m1<-lm(Finish~Age)

#Fit a nonlinear model in x (note that it is linear in beta)
lage<-log(Age)
m2<-lm(Finish~Age+lage)
#compute the mean of finishing times for all ages 
mage=tapply(Finish,Age,mean)
#Scatterplot of age and finishing times
plot(Age,Finish,ylab="Finishing Time",col="gray")
#Get the unique values of Age
A=sort(unique(Age))
#plot age versus mean of finishing times for each age level
points(A,mage,col='blue')
#plot the fitted line using the linear model
points(Age,m1$fitted.values,col='red',type="l",lwd=3)
#plot the fitted regression using the nonlinear model
points(Age,m2$fitted.values,col='green',type="l",lwd=3)


