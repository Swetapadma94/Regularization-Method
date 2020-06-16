# Libraries Needed
library(caret)
install.packages("glmnet") ## to build multiple model
library(glmnet)
library(mlbench)
library(psych)

# Data importing
data("BostonHousing")
data <- BostonHousing
View(data)
attach(data)

# Data Partition
set.seed(222)
sample
nro
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
train <- data[ind==1,]
test <- data[ind==2,]

# Custom Control Parameters
## cross validation
custom <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 5,
                       verboseIter = T)

# Linear Model
set.seed(1234)
lm <- train(medv~.,
            train,
            method='lm',
            trControl=custom)


# Results
lm$results
plot(lm$finalModel)

# Ridge Regression
set.seed(1234)
ridge <- train(medv~.,
               train,
               method='glmnet',
               tuneGrid= expand.grid(alpha=0, lambda= seq(0.001,1, length=5)),
               trControl=custom)

# Plot Results
plot(ridge)
plot(ridge$finalModel, xvar = "lambda", label = T)
plot(ridge$finalModel, xvar = 'dev', label=T)
plot(varImp(ridge, scale=T))

# Lasso Regression
set.seed(1234)
lasso <- train(medv~.,
               train,
               method='glmnet',
               tuneGrid= expand.grid(alpha=1, lambda= seq(0.001,1, length=5)),
               trControl=custom)

# Plot Results
plot(lasso)
plot(lasso$finalModel, xvar = 'lambda', label=T)

# Elastic Net Regression
set.seed(1234)
en <- train(medv~.,
            train,
            method='glmnet',
            tuneGrid= expand.grid(alpha=seq(0,1, length=10), lambda= seq(0.001,1, length=5)),
            trControl=custom)

# Plot Results
plot(en)
plot(en$finalModel, xvar = 'lambda', label=T)
plot(en$finalModel, xvar = 'dev', label=T)
plot(varImp(en))

# Compare Models
model_list <- list(LinearModel=lm, Ridge=ridge, Lasso=lasso, ElasticNet = en)
res <- resamples(model_list)

summary(res)
bwplot(res)
xyplot(res,metric = 'RMSE')


# Best Model
en$bestTune
best <- en$finalModel
coef(best, s = en$bestTune$lambda)

# Save Final Model for Later Use
saveRDS(en, "final_model.rds")
fm <- readRDS("final_model.rds")
print(fm)

# Prediction
p1 <- predict(fm, train)
sqrt(mean((train$medv-p1)^2))

p2 <- predict(fm, test)
sqrt(mean((test$medv-p2)^2))
