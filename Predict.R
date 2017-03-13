#Predict

prediction = list()
for(i in 1:nrow(newtrain))

{
  inputseq = strsplit(newtrain$eventseq[i],split = "&")
  layer_1_values = list()
  layer_1_values = append(layer_1_values, list(c(rep(0,hidden_dim))) )
  
  for(j in 1:length(inputseq))
  {
    X = event_to_array(as.numeric(inputseq[[1]][j]))
    layer1 = sigmoid( (X%*%Synapse_0) +(as.vector(as.numeric(layer_1_values[[length(layer_1_values)]]))%*%Synapse_h))
    layer2 = sigmoid(layer1%*%Synapse_1)
    layer_1_values = append(layer_1_values, list(layer1))
  }
  
  dim(layer2) = c(1,10)
  prediction = append(prediction, list(layer2))
  
  
}  
  