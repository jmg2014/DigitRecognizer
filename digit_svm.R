######################################################################################
# Approach using a Support Vector Machines
######################################################################################
library(kernlab)

#Load train data 
data<-read.csv(file="data/train.csv",head=TRUE,sep=",",stringsAsFactors=FALSE)

#Load test data
test<-read.csv(file="data/test.csv",head=TRUE,sep=",",stringsAsFactors=FALSE)


#Build the model
model<-ksvm(factor(label)~ .,data=data,kernel="rbfdot")

#Build the prediction
prediction<-predict(model,test,type="response")



# Create a data frame with two columns: ImageId & Label
x <- rep(1:nrow(test))
solution <- data.frame(ImageId=x,Label = prediction)

# The data frame should has 28000 entries
nrow(solution)

# Write the solution to a csv file called solution.csv
write.csv(solution, file = "solution.csv", row.names = FALSE)
#0.97186
