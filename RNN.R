#RNN by Zambie
#Implemented iamtrask RNN code written in python for R and customized certain parts for my specific usecase


#array of unique events, used to match input to the correct position
event.array = c(30018, 30021, 30024, 30027, 30039, 30042, 30045, 30048, 36003, 45003)

train = read.csv("train.csv",stringsAsFactors = FALSE)


#Setting model learning rate and dimensions
alpha = 0.15
input_dim = length(event.array)
hidden_dim = 16
output_dim = length(event.array)
numepochs = 10

library(sigmoid)


#Writing an event to array function which takes in as its input the event/word and returns an array with a 1 signifying where the event in event.arry lies
event_to_array <- function(x)
{
  arr = rep(0,10)
  arr[match(x,event.array)] = 1
  return(arr)
}

#Setting a seed and randomly initialising the synapse weight matrix from a uniform distibution  
set.seed(21)
Synapse_0 = matrix(runif(input_dim*hidden_dim, -1, 1),input_dim,hidden_dim)
Synapse_1 = matrix(runif(hidden_dim*output_dim, -1,1),hidden_dim,output_dim)
Synapse_h = matrix(runif(hidden_dim*hidden_dim, -1,1),hidden_dim,hidden_dim)



for(epoch in 1:numepochs)
{

  print(epoch)
  
for (i in 1:nrow(train))
{
  #Initializing the synapse weight matrix update to all zeroes
  Synapse_0_update = matrix(rep(0,input_dim*hidden_dim),input_dim,hidden_dim)
  Synapse_1_update = matrix(rep(0,hidden_dim*output_dim),hidden_dim,output_dim)
  Synapse_h_update = matrix(rep(0,hidden_dim*hidden_dim),hidden_dim,hidden_dim)
  
  #creating the input and output sequence by splitting the ne
  inputseq = strsplit(train$trainseq[i],split = "&")
  outputseq = strsplit(train$testseq[i],split = "&")
  
  layer_2_deltas = list()
  layer_1_values = list()
  layer_1_values = append(layer_1_values, list(c(rep(0,hidden_dim))) )
  overallerror = 0
  pred = c(rep(0,10))
  
  #Front Propogate
  for(j in 1:length(inputseq[[1]]))
  {
    X = event_to_array(as.numeric(inputseq[[1]][j]))
    Y = event_to_array(as.numeric(outputseq[[1]][j]))
   
    
    layer1 = sigmoid( (X%*%Synapse_0) +(as.vector(as.numeric(layer_1_values[[length(layer_1_values)]]))%*%Synapse_h))
    
    layer2 = sigmoid(layer1%*%Synapse_1)
    
    layer2_error = Y - layer2
    
    layer_2_deltas = append(layer_2_deltas,list(layer2_error*sigmoid_output_to_deravative(layer2)))

    overallerror = overallerror + sum(abs(layer2_error))
    
    pred = layer2
    
    layer_1_values = append(layer_1_values, list(layer1))
    
  }
  
  future_layer_1_delta = c(rep(0,hidden_dim))
  
  #Back propgate
  for(j in length(inputseq[[1]]):1)
  {
    X = event_to_array(as.numeric(inputseq[[1]][j]))
    layer_1 = layer_1_values[[j+1]]
    prev_layer_1 = layer_1_values[[j]]
    
    layer_2_delta = layer_2_deltas[[j]]
    
    layer_1_delta = (future_layer_1_delta%*%(t(Synapse_h))) + layer_2_delta%*%(t(Synapse_1)) * sigmoid_output_to_deravative(layer_1)
    
    dim(layer_1) = c(hidden_dim,1)
    
    Synapse_1_update = Synapse_1_update + layer_1 %*%layer_2_delta
    
    dim(prev_layer_1) = c(hidden_dim,1)
    
    Synapse_h_update = Synapse_h_update + prev_layer_1 %*% layer_1_delta
    
    dim(X) = c(input_dim,1)
    
    Synapse_0_update = Synapse_0_update + X%*%layer_1_delta
    
    future_layer_1_delta = layer_1_delta

  }
  
  Synapse_0 = Synapse_0 + (Synapse_0_update * alpha)
  Synapse_1 = Synapse_1 + (Synapse_1_update * alpha)
  Synapse_h = Synapse_h + (Synapse_h_update * alpha)
  
  if(i %% 10000 == 0)
  {
    print(overallerror)
    print(Y)
    print(pred)
  }
  
}



}