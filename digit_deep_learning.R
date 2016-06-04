######################################################################################
# Approach using Deep learning (mxnet package)
######################################################################################


######        Instalation ##############################################################

#install drat package from CRANE)

#drat:::addRepo("dmlc") in R Shell

#install.packages("mxnet") in R shell
########################################################################################


require(mxnet)
train <- read.csv("data/train.csv", header=TRUE)
test <- read.csv("data/test.csv", header=TRUE)
train <- data.matrix(train)
test <- data.matrix(test)

#pixels
train.x <- train[,-1]
#labels
train.y <- train[,1]

#The greyscale of each image falls in the range [0, 255], we can linearly transform it into [0,1]
train.x <- t(train.x/255)
test <- t(test/255)


table(train.y)

#Network Configuration

#Types of activation: {'relu', 'sigmoid', 'softrelu', 'tanh'}

#Relu:
#f(x)=max(0,x)

#Sigmoid unit :
#  f(x)=1/1+e(−x)

#Tanh unit:
#  f(x)=tanh(x)


#SoftRelu
#f(x)=log(1+ex) 


data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="tanh1", act_type="tanh")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="tanh2", act_type="tanh")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

#Training
devices <- mx.cpu()

mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     batch.end.callback=mx.callback.log.train.metric(100))


#Prediction 
preds <- predict(model, test)
#dim(preds)


pred.label <- max.col(t(preds)) - 1
#table(pred.label)

#Submission
submission <- data.frame(ImageId=1:ncol(test), Label=pred.label)
# Write the solution to a csv file called submission_deep.csv
write.csv(submission, file='submission_deep.csv', row.names=FALSE, quote=FALSE)
#0.97400