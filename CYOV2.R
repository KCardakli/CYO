## Overview
#
# In this project, we look at the US COVID deaths from Jan 2024 - Sept 23, 2023 and associated diagnoses at the time of death.  
# We fit models using the associated diagnoses predicting the number of deaths.  Once we find the best models we identify the 
# variable importance for these models.  The input to our models are the number of diagnoses for each row and the output is the 
# COVID19 deaths for that time period and geographic location.
#
# We use dataset from US Health and Human Services (HHS).  We first wrangle the data and eliminate duplicate data and rows with missing data.
# Please refer to the section "About the dataset" for further details.
#
# We then convert the data into wide format and check against the total COVID19 deaths from CDC website.
# We construct 14 models and train these models via bootsrapping and cross validation with a k value of 25.
# 
# The models that provide best RMSE (root of mean square error) values are provided by the xgbTree and rf algorithms followed by gamLoess.
# xgbTree and rf are Tree-Based Models, xgbTree is Extreme Gradient Boosting Model where as rf is Random Forest Model.
# They provide similar variable importance.  The third best model, gamLoess, combines Generalized Additive Models (GAM) with LOESS 
# (Locally Estimated Scatterplot Smoothing) and gives a different variable importance.  Please see the pdf file for details.
#
# The neural network model(nnet), provided one the worst models with an RMSE value of 2186.  We investigate to see if we can improve on this model by 
# changing the model parameters, number of units in the hidden layer and decay.  Although we improve the RMSE value significantly, it is still not 
# as accurate as xgbTree, rf, or gamLoess models.  Apparently nnet model can have only one hidden layer, and that may be limitation of this model.
#
# When looking at the results of this project, It is probably important remember that correlation doesn't mean causality; 
# some of the diagnoses could be (and are) complications of COVID19, rather than causes.  The best model gives us the 10 most 
# important variables, in the order of decreasing importance, as: respiratory failure(COVID19 itself causes respiratory failure), 
# influenza and pneumonia, malignant neoplasms (cancer), adult respiratory distress syndrome,cardiac arrhythmia, respiratory arrest, 
# chronic lower respiratory diseases, diabetes, ischemic heart disease, and heart failure.  The result seems reasonable,
# but of course much more research would have to be done to generalize.
# 
# It would be interesting to get the list of the diagnoses when the patient was diagnosed with COVID19 not at the time of death.  And of course, it would 
# be even better if we could get the diagnoses of every COVID19 patient when they got sick, not just the ones that died from it.
#
#
###
#
# About the dataset
#
###
#
# For this project we use a public dataset from US Health and Human Services (HHS):
# Conditions Contributing to COVID-19 Deaths, by State and Age, Provisional 2020-2023
# https://catalog.data.gov/dataset/conditions-contributing-to-deaths-involving-coronavirus-disease-2019-covid-19-by-age-group
# The dataset summarizes the COVID19 deaths and associated factors.
#
# I suspect the underlying full dataset is at the US Center for Disease Control (CDC):
# https://wonder.cdc.gov/mcd.html
# but this database is not public, and only available to researchers with certain conditions:
# https://wonder.cdc.gov/mcd-icd10-provisional.html
# Among the conditions is "Do not present or publish death counts of 9 or fewer or death rates based on counts of nine or fewer (in figures, graphs, maps, tables, etc.)."
# I have asked the TA of this course about the CDC dataset and I was asked not use CDC dataset since I could not provide the dataset as part of the project.
# 
# HHS dataset puts NA for values 1-9 as CDC dataset conditions dictate.
# HHS dataset also provides the national sums.
# HHS dataset provides the associated diagnoses for COVID19 deaths in each row, data is tabulated as on a per month / state basis. 
# In the State column, possible values are all US states, Washington DC, and Puerto Rico.
#
#
## Methods
# 
# We use 14 models:
# lm: Linear Regression
# glm: Generalized Linear Model
# knn: K-Nearest Neighbors
# rf: Random Forest
# gamLoess: Generalized Additive Model (GAM) combined with LOESS (Locally Estimated Scatterplot Smoothing)
# rpart: Recursive Partitioning and Regression Trees
# xgbTree: Extreme Gradient Boosting (XGBoost)
# cforest: Conditional Inference Trees
# glmnet: Regularized Generalized Linear Models (Elastic Net)
# bayesglm: Bayesian Generalized Linear Models
# pcr: Principal Component Regression
# pls: Partial Least Squares Regression
# ridge: Ridge Regression.
# nnet: Neural Network
# 
# We first start with bootstrapping k value of 25, and then we use cross validation with a k-value of 25.
#
#
#
## Results
#
# Cross validation with a k value of 25 provides the best RMSE values on the test data.
# Model     RMSE
# xgbTree   161.1663         
# rf        248.5894
# gamLoess  252.0772
#
# Bootstrapping yields similar results; the order of models do not change:
# Model     RMSE
# xgbTree   231.0553   
# rf        248.5894
# gamLoess  252.0772 
#
# 
# xgbTree variable importance gives us the model's correlated diagnoses 
# 
# Overall
# Respiratory.failure                      100.00000
# Influenza.pneumonia                       90.07434
# Malignant.neoplasms                       30.88395
# Adult.respiratory.distress.syndrome       21.18099
# Cardiac.arrhythmia                        18.13848
# Respiratory.arrest                        14.79050
# Chronic.lower.respiratory.diseases        12.53784
# Diabetes                                   5.39707
# Ischemic.heart.disease                     5.32439
# Heart.failure                              4.20999
# Hypertensive.diseases                      2.35590
# Vascular.unspecified.dementia              1.06199
# Cardiac.arrest                             0.32535
# Sepsis                                     0.29698
# Other.diseases.of.the.respiratory.system   0.22408
# Cerebrovascular.diseases                   0.17067
# Renal.failure                              0.13228
# Obesity                                    0.10758
# Alzheimer.disease                          0.03441
# Other.diseases.of.the.circulatory.system   0.00000
# 
# Variable Importance for rf model is similar:
# 
# Overall
# Influenza.pneumonia                      100.0000
# Respiratory.failure                       84.0869
# Ischemic.heart.disease                    67.4816
# Cardiac.arrhythmia                        44.0161
# Other.diseases.of.the.circulatory.system  23.0288
# Cerebrovascular.diseases                  16.9564
# Renal.failure                             16.5231
# Diabetes                                  13.6520
# Adult.respiratory.distress.syndrome       13.1184
# Respiratory.arrest                         9.0165
# Chronic.lower.respiratory.diseases         7.5493
# Heart.failure                              6.6621
# Hypertensive.diseases                      5.5820
# Cardiac.arrest                             4.9089
# Alzheimer.disease                          3.1151
# Malignant.neoplasms                        2.4882
# Other.diseases.of.the.respiratory.system   2.2472
# Sepsis                                     1.5487
# Vascular.unspecified.dementia              0.9601
# Obesity                                    0.0000
#
#
#
## Conclusions
#
# Tree based models give us the best RMSE results.  In particular xgbTree model gives the best results with 25-fold cross validation. 
# Considering the mean of the output parameter, number of COVID19 deaths, is 1828.651, and median is 1236, a RMSE of 161.1663 seems good.  
# Of course, it may be possible to improve on this RMSE value by using the base source data from CDC.
# It was also interesting to see that tree models have similar variable importance, while gamLoess has a much different variable importance.
#
## Possible Future Work
#
# It would be interesting to replicate the work with the non-public base data from CDC with the actual counts 1-9 included.  
# That would give us more data rows, and it would be more precise.  
#
# Also as mentioned before, it would be interesting to get the diagnoses of the patients when they get contract COVID19, then we can compare the 
# possible associations (possibly risk factors) between the patients that survived COVID19 and not.
#
## References
# Public dataset from US Health and Human Services (HHS):
# Conditions Contributing to COVID-19 Deaths, by State and Age, Provisional 2020-2023
# https://catalog.data.gov/dataset/conditions-contributing-to-deaths-involving-coronavirus-disease-2019-covid-19-by-age-group
#
#
#
#########################################################################
#
# Please uncomment these lines to install packages as needed in your system:
#
#install.packages("dplyr")
#install.packages("data.table")
#install.packages("readr")
#install.packages("caret")
#install.packages("tidyr")
#install.packages("xgboost")


library("dplyr")
library("data.table")
library("readr")
library("caret")
library("tidyr")
library("xgboost")

options(max.print = 500)
options(warn = -1)

#########################################################################
# !!!!
# IMPORTANT: Please modify the directory to your working directory as needed, 
# all 4 files should be in your working directory of your RStudio.
#
print("working directory: ")
setwd("/Users/kccardakli/Documents/R/projects/Capstone Project/CYO")
getwd()

options(timeout = 1200)

# Specifying the URL of the uncompressed file from CDC
# If this fails for you, you can download the zipped file from GitHub, and unzip it manually: 
# https://github.com/KCardakli/CYO/blob/4c86003ad5095c7baceed361753182a35b30a439/Conditions_Contributing_to_COVID-19_Deaths__by_State_and_Age__Provisional_2020-2023.csv.zip
# or the uncompressed file from Google drive:
# https://drive.google.com/file/d/1WQt3y2N-xiEfEiuI4VuExcexpjjchNe2/view?usp=sharing
# R-studio fails when large files are downloaded from GitHub or Google drive
url <- "https://data.cdc.gov/api/views/hk9y-quqm/rows.csv?accessType=DOWNLOAD"

# Name of the downloaded file 
destfile <- "Conditions_Contributing_to_COVID-19_Deaths__by_State_and_Age__Provisional_2020-2023.csv" 

# Downloading the file 
if(!file.exists(destfile))
  download.file(url, destfile, method = "libcurl")

# reading the HHS dataset
data <- read.csv("./Conditions_Contributing_to_COVID-19_Deaths__by_State_and_Age__Provisional_2020-2023.csv") 

# Let's check the data for one month and one state
# We will use the Age.Group == "All Ages" as it minimizes the effect of putting NA for values 1-9 as explained above.
#
temp <- data |> filter(Month == 8 & Year == 2020 & State == "Alabama" & Age.Group == "All Ages") |> select(-Data.As.Of, -Start.Date, -End.Date, -ICD10_codes) 
write_csv(temp, "temp.csv")

#########################################################################
# Data Wrangling
#


# Eliminate the nationwide data as they replicate the statewide data by summation.
# We don't want to double count the data.
# We will use the monthly data.
# Eliminate the rows that have "" or NA for the COVID.19.Deaths, this comes from the CDC source data where 
#  "one or more data cells have counts between 1-9 and have been suppressed in accordance with NCHS confidentiality standards".
#  this elimination introduces some error, solution would be to use the CDC dataset, but it is not public.  
#  We also eliminate  "All other conditions and causes (residual)", "Intentional and unintentional injury, poisoning, and other adverse events"
#  as these are catchall phrases and are not specific diagnoses.
#  We are using the COVID.19.Deaths and not Number.of.Mentions as some conditions are mentioned multiple times in death certificates
#  under reason for death and contributing factors, so they could be counted more than once.
data <- data |> filter(data$Group == "By Month" & 
                       data$COVID.19.Deaths != "" & 
                       State != "United States" &
                       !is.na(data$COVID.19.Deaths)  & 
                       Condition != 'All other conditions and causes (residual)'  & 
                       Condition != 'Intentional and unintentional injury, poisoning, and other adverse events'  & 
                       Age.Group == 'All Ages') |> select(Year, Month, State, COVID.19.Deaths, Condition)
str(data)

# Let's check the total number of COVID deaths in the dataset
# It is inline with other resources from CDC (1.14M):
# https://www.cdc.gov/nchs/nvss/vsrr/covid19/index.htm
# 
t <- data |> filter(Condition == "COVID-19" & !is.na(COVID.19.Deaths)) 
print("Total COVID deaths in the dataset  01/01/2020 - 09/23/2023:")
sum(t$COVID.19.Deaths)


# We now rename the fields that will become columns so that they are easily readable and can be accessed without using `` characters
data <- data |>
  mutate(Condition = gsub(",", "", Condition)) |>
  mutate(Condition = gsub(" and ", " ", Condition)) |>
  mutate(Condition = gsub('-', "", Condition)) |>
  mutate(Condition = gsub("  ", " ", Condition)) |>
  mutate(Condition = gsub(" ", ".", Condition)) 
  
#These are the unique conditions
unique(data$Condition)

# Form the wide data with "Condition"s are columns   
wide_data <- data |>
  pivot_wider(id_cols = c(Year, Month, State), names_from = Condition, values_from = c(COVID.19.Deaths)) 
str(wide_data)
wide_data |> print(, n = 25)

write_csv(wide_data, "wide_data_export.csv")

# We have lots of values with NA values
# We could assume some values, but that would introduce significant errors to the models.
# Instead, let's delete the rows that have NA values, this will reduce the rows in our dataset
# but there will be no errors introduced by the NA value due to counts of 1-9.
wide_data <- wide_data[(complete.cases(wide_data) == TRUE), ]

# Now lets also delete the rows that have 0 COVID-19 deaths.
# This was common at the beginning of 2020.
wide_data <- wide_data |> filter(`COVID19` > 0)

# Change the Month column to be a linear parameter as it gives us a better parameter for the models, 
#   no need for the Year parameter afterwards
# Experimented with using the month as a parameter as well
#  But it did not provide much improvement, so eliminated the Month parameter later on..
wide_data <- wide_data |> mutate(Month = (Year - 2020)*12 + Month) |> select(-Year, -State, -Month)

#This is our final dataset
str(wide_data)
write_csv(wide_data, "wide_data_export.csv")

print('Mean of our output Parameter, COVID19:')
mean(wide_data$COVID19)

print('Median of our output Parameter, COVID19:')
median(wide_data$COVID19)

#
# Let's prepare our training and test datasets
# Final hold-out test set will be 10% of the data
# We use the naming convention as in the previous project
#
# Set the seed for reproducibility
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier

test_index <- createDataPartition(y = wide_data$COVID19, times = 1, p = 0.1, list = FALSE)
edx <- wide_data[-test_index,]
final_holdout_test <- wide_data[test_index,]


#########################################################################
#
# Comparing models with bootstrapping method with a value of 25
# On my computer this subsection takes 2 minutes 50 seconds 
#

# Define the trainControl with bootstrapping
train_control <- trainControl(
  method = "boot",     # Bootstrapping method
  number = 25,         # Number of bootstrap samples
  verboseIter = FALSE # Set to TRUE to see the progress
)

  
# List of different model types to run
models_list <- c("lm", "glm", "knn", "rf", "gamLoess", "rpart", "xgbTree", "cforest", "glmnet", "bayesglm", "pcr", "pls", "ridge", "nnet")

# Create empty lists to store the results
boot_model_results <- list()
boot_rmse <- list()

# Loop over the list of models and train each one
# suppress the warnings about deprecated functions
# and nnet training messages
for (model_type in models_list) invisible(capture.output({
  # to be able to replicate the results
  set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
  # set.seed(1) # if using R 3.5 or earlier
  
  model <- train(
    COVID19 ~ .,              # COVID19 variable vs all predictors
    data = edx,               # Training dataset
    method = model_type,      # Current model from the list
    trControl = train_control # Use the defined trainControl
    
  )
  
  # Predict on the final_holdout_test dataset
  predictions <- predict(model, newdata = final_holdout_test)
  
  boot_rmse[[model_type]] <- sqrt(mean((predictions - final_holdout_test$COVID19)^2))
  
  # Store the result in the list
  boot_model_results[[model_type]] <- model
}))

# Access results of each model
boot_rmse

# Let's look at the best models and worst models

boot_top_3_rmse <- sort(unlist(boot_rmse))[1:3]
boot_top_3_rmse

boot_worst_3_rmse <- sort(unlist(boot_rmse), decreasing = TRUE)[1:3]
boot_worst_3_rmse

# Extract the variable importance for top 3 models
for (model_name in names(boot_top_3_rmse)) {
  cat("Variable Importance for model:", model_name, "\n")
  var_importance <- varImp(boot_model_results[[model_name]])
  print(var_importance)
  cat("\n")
}


#########################################################################
#
# Comparing models with cross validation  with a k value of 25
# On my computer this subsection takes 2 minutes 50 seconds 
#


# Define the trainControl with cross validation method
train_control <- trainControl(
  method = "cv",       # cross validation method
  number = 25,         # Number of bootstrap samples
  verboseIter = FALSE  # Set to TRUE to see the progress
)

# Create empty lists to store the results
cv_model_results <- list()
cv_rmse <- list()

# Loop over the list of models and train each one
# suppress the warnings about deprecated functions
# and nnet training messsages
for (model_type in models_list) invisible(capture.output({
  # to be able to replicate the results
  set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
  # set.seed(1) # if using R 3.5 or earlier
  model <- train(
    COVID19 ~ .,              # COVID19 variable vs all predictors
    data = edx,               # Training dataset
    method = model_type,      # Current model from the list
    trControl = train_control # Use the defined trainControl
  )
  
  # Predict on the final_holdout_test dataset
  predictions <- predict(model, newdata = final_holdout_test)
  
  cv_rmse[[model_type]] <- sqrt(mean((predictions - final_holdout_test$COVID19)^2))
  
  # Store the result in the list
  cv_model_results[[model_type]] <- model
}))

# Access results of each model:
cv_rmse

# Let's look at the best models and worst models
cv_top_3_rmse <- sort(unlist(cv_rmse))[1:3]
cv_top_3_rmse

cv_worst_3_rmse <- sort(unlist(cv_rmse), decreasing = TRUE)[1:3]
cv_worst_3_rmse

# Extract the variable importance for top 3 models
for (model_name in names(cv_top_3_rmse)) {
  cat("Variable Importance for model:", model_name, "\n")
  var_importance <- varImp(cv_model_results[[model_name]])
  print(var_importance)
  cat("\n")
}

#########################################################################
# The best models each way of training are 
# xgbTree,rf,gamLoess 
# The worst one is nnet with rmse value of 2186.6641
# 


#########################################################################
# Is there a way to optimize the nnet model?
# In this section, we change the number of units in the hidden layer 
# and delay parameters
# This section takes 2 minutes 55 seconds on my computer.

# Define the control function with cross-validation
train_control <- trainControl(method = "cv", 
                              number = 10)

# Define a grid of parameters to tune
tune_grid <- expand.grid(
  size = c(3, 4, 5, 6, 7, 8),    # Number of units in the hidden layer
  decay = c(0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16)  # Regularization parameter to avoid overfitting
)
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier

# Train the model using nnet and parameter tuning
# suppress training messages coming from nnet model
model <- train(
  COVID19 ~ .,              # Formula: COVID19 variable vs all predictors
  data = edx, 
  method = "nnet", 
  trControl = train_control,
  tuneGrid = tune_grid,
  linout = TRUE,        # For regression, use linout = TRUE
  trace = FALSE,        # Avoid printing during training
  maxit = 1000          # Maximum number of iterations
)

# View the best model
print(model$finalModel)


# Predict on the final_holdout_test dataset
predictions <- predict(model$finalModel, newdata = final_holdout_test)

sqrt(mean((predictions - final_holdout_test$COVID19)^2))

# rmse has improved significantly, 1492.863
# but still it is much worse than our best model xgbTree: 161
# apparently nnet can only have one layer of hidden layer.
# To have multiple hidden layers, keras package from Google can be used apparently.


