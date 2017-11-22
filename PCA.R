# Principal Component Analysis
#-----------------------------

library(caret)
library(e1071)

# Read in the data
dataset = read.csv('C:/Users/Amit/Desktop/Udemy course/Data/Wine.csv')


# Create the training and test sets
set.seed(123)
split = createDataPartition(dataset$Customer_Segment, p=0.8, list=FALSE)
train = dataset[split,]
test  = dataset[-split,]

# Feature scaling
train[,-14] = scale(train[,-14])
test[,-14]  = scale(test[,-14])

# Do PCA for creating two factors
pca = preProcess(train[,-14], 
                 method='pca',
                 pcaComp = 2)

train = predict(pca, train)   # column orders change. The DV becomes the first variable
test =  predict(pca, test)

# Just rearranging the columns
train = train[,c(2,3,1)]
test  = test[,c(2,3,1)]

# Create a logistic rgression model on the reduced data
modSVM = svm(Customer_Segment~.,data=train,
             type='C-classification',
             kernel='linear')


# Predict the outcomes using the model
y_pred = predict(modSVM, test[,-3])


# Evaluate the model
confusionMatrix(y_pred, test[,3])



# Visualize the decision boundaries for training set
set = train

X1 = seq(from=min(set[,1])-1, to=max(set[,1]+1), by=0.02)
X2 = seq(from=min(set[,2])-1, to=max(set[,2]+1), by=0.02)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(modSVM, grid_set)

plot(set[,-3],
     main = 'SVM after Principal Component Analysis',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))

#contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add=TRUE)   # this is optional

points(grid_set, pch='.',col=ifelse(y_grid==2,'deepskyblue',ifelse(y_grid==1,'springgreen3','tomato')))
points(set, pch=21, bg=ifelse(set[,3]==2,'blue3', ifelse(set[,3]==1, 'green4','red3')))




# Visualize the decision boundaries for test set
set = test

X1 = seq(from=min(set[,1])-1, to=max(set[,1]+1), by=0.02)
X2 = seq(from=min(set[,2])-1, to=max(set[,2]+1), by=0.02)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(modSVM, grid_set)

plot(set[,-3],
     main = 'SVM after Principal Component Analysis',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))

#contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add=TRUE)   # this is optional

points(grid_set, pch='.',col=ifelse(y_grid==2,'deepskyblue',ifelse(y_grid==1,'springgreen3','tomato')))
points(set, pch=21, bg=ifelse(set[,3]==2,'blue3', ifelse(set[,3]==1, 'green4','red3')))

