
## Replace NA with the mean

data1_original = read.csv("/Users/kenialopez/Documents/P5/DM/pivot_df.csv", header = TRUE)
data1 <- as.data.frame(data1_original)


data1_reduced <- subset(data1, select = -c(id, time, date, hour))
col_names <- colnames(data1_reduced)

for (col in col_names)
  data1_reduced[[col]][which(is.na(data1_reduced[[col]]))] <- mean(data1_reduced[[col]], na.rm = TRUE)

data1[names(data1_reduced)] <- data1_reduced
data1[c("mood")] <- round(data1[c("mood")])
data1[c("circumplex.arousal")] <- round(data1[c("circumplex.arousal")])
data1[c("circumplex.valence")] <- round(data1[c("circumplex.valence")])



## Replace NA by sampling 
set.seed(42)
data2_original <- read.csv("/Users/kenialopez/Documents/P5/DM/pivot_df.csv", header = TRUE)
data2 <- as.data.frame(data2_original)

data2_reduced <- subset(data2, select = -c(id, time, date, hour))
col_names_df2 <- colnames(data2_reduced)

for (col in col_names_df2){
  col_no_na <- data2_reduced[[col]][!is.na(data2_reduced[[col]])] # Extract non-NA values from column
  num_NA <- sum(is.na(data2_reduced[[col]])) # Count the number of NA values in column
  new_sample <- sample(col_no_na, size=num_NA, replace = TRUE) # Sample from col without NA to create new_sample of samples
  data2_reduced[[col]][is.na(data2_reduced[[col]])] <- new_sample # Replace NA values in column with values from new_sample
}

data2[names(data2_reduced)] <- data2_reduced


##Check for Multicollinearity

variables <- subset(data2, select = -c(id, time, date, hour))
variable_variance <- sapply(variables, var)
variance_df <- data.frame(variable_variance)                     
corr_variables <- cor(data2_reduced, method = "spearman")


# Results: No variance in variable "call" and "sms" (variance = 0). 
# If one of the variables has zero variability, the covariance or correlation with other variables will be undefined,
# resulting in NA values.

#Correlation between variable is small so there is no correlation between all the variables.



## Do we remove SMS and CALL?

## MODEL 1

y1 <- data1_reduced$mood

model1 <- lm(y1 ~ activity + appCat.builtin + appCat.communication + appCat.entertainment + appCat.finance + appCat.game +
                 appCat.office + appCat.other + appCat.social + appCat.travel + appCat.unknown + appCat.utilities +
                 appCat.weather + circumplex.arousal + circumplex.valence + screen, data = data1_reduced)

summary(model1) #activity, circumplex.arousal and circumplex.valence are significant. Multiple R-squared:  0.4827


## MODEL 2

y2 <- data2_reduced$mood

model2 <- lm(y2 ~ activity + appCat.builtin + appCat.communication + appCat.entertainment + appCat.finance + appCat.game +
              appCat.office + appCat.other + appCat.social + appCat.travel + appCat.unknown + appCat.utilities +
              appCat.weather + circumplex.arousal + circumplex.valence + screen, data = data2_reduced)

summary(model2) #circumplex.arousal and circumplex.valence are significant. Multiple R-squared:  0.0001332

#model1 seems to be better than model2 (Multiple R-squared. The higher the better)


## STEP UP METHOD
install.packages("olsrr")

result1_up <- ols_step_forward_p(model1, p_val = 0.05)
  # circumplex.valence + circumplex.arousal + activity  are significant.
  # R-Squared: 0.483
  # RMSE: 0.093
  # MSE: 0.009
result2_up <- ols_step_forward_p(model2, p_val = 0.05)
  # circumplex.valence + circumplex.arousal are significant
  # R-Squared: 0.000
  # RMSE: 1.031
  # MSE: 1.064


features_m1_up <- result1_up$metrics
features_m2_up <- result2_up$metrics


## STEP DOWN METHOD

result1_down <- ols_step_backward_p(model1, p_val = 0.05) 
  # circumplex.valence + circumplex.arousal + activity  are significant.
  # R-Squared: 0.483     
  # RMSE: 0.093
  # MSE: 0.009
result2_down <- ols_step_backward_p(model2, p_val = 0.05)
  # circumplex.valence + circumplex.arousal + activity  are significant.
  # R-Squared: 0.000
  # RMSE: 1.031
  # MSE: 1.064

features_m1_down <- result1_down$metrics
features_m2_down <- result2_down$metrics


## EXHAUSTIVE FEATURE SELECTION 

install.packages("ExhaustiveSearch")

efs <- ExhaustiveSearch(model1, data = data1_reduced, family=NULL)
  # Top features: activity + circumplex.arousal + circumplex.valence

efs_2 <- ExhaustiveSearch(model2, data = data2_reduced, family=NULL)
  # Top features: appCat.finance + circumplex.arousal + circumplex.valence
  # Top features: circumplex.arousal + circumplex.valence


## LASSO feature selection 

library(glmnet)


x <- as.matrix(subset(data1, select = -c(mood, time, hour, id, date)))
Y=data1[, 'mood']

X <- scale(x) #normalization of predictor

set.seed(222)

train=sample(1:nrow(X),0.67*nrow(X)) # train by using 2/3 of the x rows 
X.train=X[train,]; Y.train=Y[train]  # data to train
X.test=X[-train,]; Y.test = Y[-train]


## LASSO
lasso.model=glmnet(X.train,Y.train,alpha=1) 
lasso.cv=cv.glmnet(X.train,Y.train,alpha=1,type.measure="mse")

plot(lasso.model,label=T,xvar="lambda") 
plot(lasso.cv) 

lambda.min=lasso.cv$lambda.min; lambda.1se=lasso.cv$lambda.1se; 
lambda.min; lambda.1se # best lambda by cross validation
coef(lasso.model,s=lasso.cv$lambda.min) # Activity, circumplex.arousal and circumplex.valence are relevant
coef(lasso.model,s=lasso.cv$lambda.1se) 




# RIDGE REGRESSION

X_drop <- as.matrix(subset(data1, select = -c(mood, time, hour, id, date, sms, call)))

set.seed(222)

train_drop=sample(1:nrow(X_drop),0.67*nrow(X_drop)) # train by using 2/3 of the x rows 
X_drop.train=X_drop[train_drop,]; Y.train=Y[train_drop]  # data to train
X_drop.test=X_drop[-train_drop,]; Y.test = Y[-train_drop]


ridge.model=glmnet(X_drop.train,Y.train,alpha=0) 
ridge.cv =cv.glmnet(X_drop.train,Y.train,alpha=0,type.measure="mse")

plot(ridge.model,label=T,xvar="lambda") 
plot(ridge.cv) 

lambda.min=ridge.cv$lambda.min; lambda.1se=ridge.cv$lambda.1se; 
lambda.min; lambda.1se # best lambda by cross validation
coef(ridge.model,s=ridge.cv$lambda.min) # All are relevant
coef(ridge.model,s=ridge.cv$lambda.1se) 


# ELASTIC NET







# ORDINAL LOGISTIC REGRESSION (RESPONSE VARIABLE IS ABOUT LEVEL OF MOOD)
# (proportional odds logistic regression)

install.packages("packagename")

new_data1_predictors <- (subset(data1, select = -c(mood, time, hour, id, date)))

new_data1_predictors_scaled <- scale(new_data1_predictors)
d <- new_data1_predictors_scaled

y_factor <- as.factor(data1$mood)
model3 <- polr(y_factor ~ activity + appCat.builtin + appCat.communication + appCat.entertainment + 
                 appCat.finance + appCat.game + appCat.office + appCat.other + appCat.social + 
                 appCat.travel + appCat.unknown + appCat.utilities + appCat.weather + circumplex.arousal +
                 circumplex.valence + screen, data = d, Hess=TRUE)



## store table
summary_table <- coef(summary(model3))

## calculate and store p values (Obtain the significance of coefficients)
pval <- pnorm(abs(summary_table[, "t value"]), lower.tail = FALSE) * 2

## combined table
summary_table <- cbind(summary_table, "p value" = round(pval,3))

#signifcant variables: activity, utilities, weather, arousal, valence.







#----------------------------------

## Check normality of variables
# --> shapiro.test(data2_reduced$activity). ERROR: sample size must be between 3 and 5000
#shapiro.test(data2_reduced$activity[0:5000]) # Activity data is not normal

## MIXED MODELS --> an individual has many observation. We have "longitudinal studies"

# see EDDA code page 311/361

## Interaction between variables - data2




