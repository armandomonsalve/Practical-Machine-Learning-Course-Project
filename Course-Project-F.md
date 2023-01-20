Practical Machine Learning - Course Project
================
Armando Monsalve
2023-01-02

## Executive Summary

In this report, we are describing how a machine learning algorithm was
built to predict the manner in which 6 participants did an exercise.
Using cross validation, specifying what the expected out of sample error
was, and why some choices were made. This prediction model was also used
to predict 20 different test cases.

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, my goal will
be to use data from accelerometers on the belt, forearm, arm, and
dumbbell of 6 participants.

## Loading Data

### Libraries

``` r
library(dplyr)
library(caret)
library(rattle)
```

### Data: Trainning set

``` r
training_set <-
  read.csv(
    file = "C:/Users/ajmon/OneDrive/AJMR/Armando Monsalve/DATA SCIENCE/COURSES/Johns Hopkins University Data Science Course/8 Practical Machine Learning/Week 4/Course Project/pml-training.csv"
    )
```

``` r
training_set %>%
  dim()
```

    ## [1] 19622   160

### Test set

``` r
testing_set <-
  read.csv(
    file = "C:/Users/ajmon/OneDrive/AJMR/Armando Monsalve/DATA SCIENCE/COURSES/Johns Hopkins University Data Science Course/8 Practical Machine Learning/Week 4/Course Project/pml-testing.csv"
    )
```

``` r
testing_set %>%
  dim()
```

    ## [1]  20 160

### Cleaning Training Set

``` r
# Identifying columns that have more than 90% of their data points null
empty_cols <-
  which(
    colSums(is.na(training_set) | training_set == "") > 0.9 * dim(training_set)[1]
  )

# Removing mostly-null and unnecessary columns
training_clean <-
  training_set[, -empty_cols]

training_clean <-
  training_clean[, -c(1:7)]
```

Let’s check how many variables we have left in my training set:

``` r
training_clean %>%
  dim()
```

    ## [1] 19622    53

Now, let’s take a quick peek at the variables:

``` r
training_clean %>%
  str()
```

    ## 'data.frame':    19622 obs. of  53 variables:
    ##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
    ##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
    ##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
    ##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
    ##  $ gyros_belt_x        : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
    ##  $ gyros_belt_y        : num  0 0 0 0 0.02 0 0 0 0 0 ...
    ##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
    ##  $ accel_belt_x        : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
    ##  $ accel_belt_y        : int  4 4 5 3 2 4 3 4 2 4 ...
    ##  $ accel_belt_z        : int  22 22 23 21 24 21 21 21 24 22 ...
    ##  $ magnet_belt_x       : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
    ##  $ magnet_belt_y       : int  599 608 600 604 600 603 599 603 602 609 ...
    ##  $ magnet_belt_z       : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
    ##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
    ##  $ pitch_arm           : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
    ##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
    ##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
    ##  $ gyros_arm_x         : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
    ##  $ gyros_arm_y         : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
    ##  $ gyros_arm_z         : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
    ##  $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
    ##  $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
    ##  $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
    ##  $ magnet_arm_x        : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
    ##  $ magnet_arm_y        : int  337 337 344 344 337 342 336 338 341 334 ...
    ##  $ magnet_arm_z        : int  516 513 513 512 506 513 509 510 518 516 ...
    ##  $ roll_dumbbell       : num  13.1 13.1 12.9 13.4 13.4 ...
    ##  $ pitch_dumbbell      : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
    ##  $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
    ##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
    ##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
    ##  $ gyros_dumbbell_z    : num  0 0 0 -0.02 0 0 0 0 0 0 ...
    ##  $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
    ##  $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
    ##  $ accel_dumbbell_z    : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
    ##  $ magnet_dumbbell_x   : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
    ##  $ magnet_dumbbell_y   : int  293 296 298 303 292 294 295 300 292 291 ...
    ##  $ magnet_dumbbell_z   : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
    ##  $ roll_forearm        : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
    ##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
    ##  $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
    ##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
    ##  $ gyros_forearm_x     : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
    ##  $ gyros_forearm_y     : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
    ##  $ gyros_forearm_z     : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
    ##  $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
    ##  $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
    ##  $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
    ##  $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
    ##  $ magnet_forearm_y    : num  654 661 658 658 655 660 659 660 653 656 ...
    ##  $ magnet_forearm_z    : num  476 473 469 469 473 478 470 474 476 473 ...
    ##  $ classe              : chr  "A" "A" "A" "A" ...

Lastly, we can check if we have variables in which we don’t have much
variation. This since if we have a column wherein most of its data
points are the same value, it won’t serve us well for our prediction.

``` r
nearZeroVar(
  training_clean
)
```

    ## integer(0)

As shown, we don’t have any variables that have near-zero variance; so
we keep them all for our prediction.

## Validation Partition

Since we need to test our models for out-of-sample error, I’ll go ahead
and split the set in my final training and validation sets:

``` r
inTrain <-
  createDataPartition(
    training_clean$classe
    , p = 0.7
    , list = FALSE
  )

training <-
  training_clean[inTrain,]

validation <-
  training_clean[-inTrain,]

# Checking how many rows my 2 sets have:
training %>%
  dim()
```

    ## [1] 13737    53

``` r
validation %>%
  dim()
```

    ## [1] 5885   53

## Building and Testing Predictive Models

I’ll build multiple models in order to find the best (accuracy-wise).
The models I’ll use are: Decision Trees, Random Forest and Gradient
Boosted Trees.

### Cross-validation

For each model I’ll use cross-validation, so they don’t get overfitted
results. I’ll implement 3-fold cross-validation:

``` r
control <-
  trainControl(
    method = "cv"
    , number = 3
    , verboseIter = F
  )
```

### Decision Tree Model

``` r
set.seed(12345)
model_tree <-
  train(
    classe ~ .
    , method = "rpart"
    , data = training
    , trControl = control
    , tuneLength = 4
  )

# Resulting tree:
fancyRpartPlot(
  model_tree$finalModel
)
```

![](Course-Project-F_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

Results (Confusion Matrix and Accuracy):

``` r
pred_tree <-
  predict(
    model_tree
    , validation
  )

confusionMatrix(
  pred_tree
  , as.factor(validation$classe)
)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1514  488  489  440  148
    ##          B   25  352   33  180  139
    ##          C   92  149  419  126  127
    ##          D   38  150   85  218  174
    ##          E    5    0    0    0  494
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5093          
    ##                  95% CI : (0.4964, 0.5221)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3594          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9044  0.30904   0.4084  0.22614  0.45656
    ## Specificity            0.6284  0.92056   0.8983  0.90916  0.99896
    ## Pos Pred Value         0.4917  0.48285   0.4589  0.32782  0.98998
    ## Neg Pred Value         0.9430  0.84736   0.8779  0.85709  0.89083
    ## Prevalence             0.2845  0.19354   0.1743  0.16381  0.18386
    ## Detection Rate         0.2573  0.05981   0.0712  0.03704  0.08394
    ## Detection Prevalence   0.5232  0.12387   0.1551  0.11300  0.08479
    ## Balanced Accuracy      0.7664  0.61480   0.6534  0.56765  0.72776

Decision Tree Accuracy = 52.8%

### Random Forest

Model and results (confusion matrix and accuracy):

``` r
set.seed(12345)
model_rf <-
  train(
    classe ~ .
    , method = "rf"
    , data = training
    , trControl = control
    , tuneLength = 5
  )

pred_rf <-
  predict(
    model_rf
    , validation
  )

# Confusion Matrix and Accuracy
confusionMatrix(
  pred_rf
  , as.factor(validation$classe)
)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    5    0    0    0
    ##          B    0 1129    9    0    0
    ##          C    0    5 1016    7    0
    ##          D    0    0    1  956    4
    ##          E    1    0    0    1 1078
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9944          
    ##                  95% CI : (0.9921, 0.9961)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9929          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9912   0.9903   0.9917   0.9963
    ## Specificity            0.9988   0.9981   0.9975   0.9990   0.9996
    ## Pos Pred Value         0.9970   0.9921   0.9883   0.9948   0.9981
    ## Neg Pred Value         0.9998   0.9979   0.9979   0.9984   0.9992
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1918   0.1726   0.1624   0.1832
    ## Detection Prevalence   0.2851   0.1934   0.1747   0.1633   0.1835
    ## Balanced Accuracy      0.9991   0.9947   0.9939   0.9953   0.9979

Random Forest Accuracy = 99.6%

Let’s take a look at the most important variables in the model:

``` r
varImp(
 model_rf
)
```

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 52)
    ## 
    ##                      Overall
    ## roll_belt             100.00
    ## yaw_belt               66.71
    ## pitch_forearm          64.14
    ## magnet_dumbbell_z      53.28
    ## magnet_dumbbell_y      50.82
    ## pitch_belt             49.80
    ## roll_forearm           48.89
    ## magnet_dumbbell_x      27.06
    ## roll_dumbbell          25.63
    ## accel_dumbbell_y       25.01
    ## magnet_belt_z          23.39
    ## accel_belt_z           23.09
    ## magnet_belt_y          20.59
    ## accel_forearm_x        18.13
    ## accel_dumbbell_z       16.99
    ## magnet_forearm_z       16.69
    ## total_accel_dumbbell   16.68
    ## roll_arm               16.57
    ## gyros_belt_z           16.07
    ## yaw_arm                12.82

### Gradient Boosted Model

Model:

``` r
set.seed(12345)
model_gbm <-
  train(
    classe ~ .
    , method = "gbm"
    , data = training
    , trControl = control
  )
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1292
    ##      2        1.5223             nan     0.1000    0.0879
    ##      3        1.4624             nan     0.1000    0.0684
    ##      4        1.4176             nan     0.1000    0.0513
    ##      5        1.3830             nan     0.1000    0.0509
    ##      6        1.3496             nan     0.1000    0.0449
    ##      7        1.3210             nan     0.1000    0.0377
    ##      8        1.2966             nan     0.1000    0.0342
    ##      9        1.2740             nan     0.1000    0.0327
    ##     10        1.2520             nan     0.1000    0.0285
    ##     20        1.0960             nan     0.1000    0.0195
    ##     40        0.9245             nan     0.1000    0.0081
    ##     60        0.8176             nan     0.1000    0.0048
    ##     80        0.7385             nan     0.1000    0.0031
    ##    100        0.6753             nan     0.1000    0.0043
    ##    120        0.6217             nan     0.1000    0.0021
    ##    140        0.5790             nan     0.1000    0.0029
    ##    150        0.5579             nan     0.1000    0.0024
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1839
    ##      2        1.4903             nan     0.1000    0.1303
    ##      3        1.4068             nan     0.1000    0.1050
    ##      4        1.3384             nan     0.1000    0.0932
    ##      5        1.2790             nan     0.1000    0.0682
    ##      6        1.2351             nan     0.1000    0.0628
    ##      7        1.1940             nan     0.1000    0.0588
    ##      8        1.1566             nan     0.1000    0.0547
    ##      9        1.1227             nan     0.1000    0.0491
    ##     10        1.0912             nan     0.1000    0.0383
    ##     20        0.8880             nan     0.1000    0.0205
    ##     40        0.6782             nan     0.1000    0.0142
    ##     60        0.5515             nan     0.1000    0.0065
    ##     80        0.4613             nan     0.1000    0.0044
    ##    100        0.3958             nan     0.1000    0.0035
    ##    120        0.3427             nan     0.1000    0.0029
    ##    140        0.3026             nan     0.1000    0.0020
    ##    150        0.2849             nan     0.1000    0.0021
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2274
    ##      2        1.4606             nan     0.1000    0.1576
    ##      3        1.3572             nan     0.1000    0.1295
    ##      4        1.2752             nan     0.1000    0.1058
    ##      5        1.2062             nan     0.1000    0.0849
    ##      6        1.1519             nan     0.1000    0.0804
    ##      7        1.1007             nan     0.1000    0.0774
    ##      8        1.0519             nan     0.1000    0.0633
    ##      9        1.0120             nan     0.1000    0.0505
    ##     10        0.9783             nan     0.1000    0.0581
    ##     20        0.7517             nan     0.1000    0.0250
    ##     40        0.5251             nan     0.1000    0.0123
    ##     60        0.4019             nan     0.1000    0.0060
    ##     80        0.3178             nan     0.1000    0.0042
    ##    100        0.2600             nan     0.1000    0.0032
    ##    120        0.2186             nan     0.1000    0.0013
    ##    140        0.1869             nan     0.1000    0.0012
    ##    150        0.1721             nan     0.1000    0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1258
    ##      2        1.5233             nan     0.1000    0.0845
    ##      3        1.4661             nan     0.1000    0.0675
    ##      4        1.4226             nan     0.1000    0.0529
    ##      5        1.3884             nan     0.1000    0.0527
    ##      6        1.3550             nan     0.1000    0.0426
    ##      7        1.3273             nan     0.1000    0.0353
    ##      8        1.3038             nan     0.1000    0.0338
    ##      9        1.2812             nan     0.1000    0.0335
    ##     10        1.2588             nan     0.1000    0.0301
    ##     20        1.1012             nan     0.1000    0.0173
    ##     40        0.9264             nan     0.1000    0.0070
    ##     60        0.8193             nan     0.1000    0.0044
    ##     80        0.7374             nan     0.1000    0.0039
    ##    100        0.6757             nan     0.1000    0.0031
    ##    120        0.6246             nan     0.1000    0.0018
    ##    140        0.5812             nan     0.1000    0.0019
    ##    150        0.5604             nan     0.1000    0.0023
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1783
    ##      2        1.4925             nan     0.1000    0.1381
    ##      3        1.4043             nan     0.1000    0.1027
    ##      4        1.3378             nan     0.1000    0.0871
    ##      5        1.2826             nan     0.1000    0.0721
    ##      6        1.2363             nan     0.1000    0.0696
    ##      7        1.1924             nan     0.1000    0.0548
    ##      8        1.1576             nan     0.1000    0.0552
    ##      9        1.1226             nan     0.1000    0.0414
    ##     10        1.0952             nan     0.1000    0.0424
    ##     20        0.8891             nan     0.1000    0.0247
    ##     40        0.6816             nan     0.1000    0.0096
    ##     60        0.5487             nan     0.1000    0.0083
    ##     80        0.4614             nan     0.1000    0.0068
    ##    100        0.3941             nan     0.1000    0.0033
    ##    120        0.3428             nan     0.1000    0.0014
    ##    140        0.3003             nan     0.1000    0.0011
    ##    150        0.2842             nan     0.1000    0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2357
    ##      2        1.4622             nan     0.1000    0.1653
    ##      3        1.3578             nan     0.1000    0.1249
    ##      4        1.2788             nan     0.1000    0.1008
    ##      5        1.2144             nan     0.1000    0.0870
    ##      6        1.1576             nan     0.1000    0.0651
    ##      7        1.1153             nan     0.1000    0.0695
    ##      8        1.0711             nan     0.1000    0.0721
    ##      9        1.0263             nan     0.1000    0.0569
    ##     10        0.9899             nan     0.1000    0.0606
    ##     20        0.7625             nan     0.1000    0.0224
    ##     40        0.5373             nan     0.1000    0.0105
    ##     60        0.4076             nan     0.1000    0.0064
    ##     80        0.3282             nan     0.1000    0.0048
    ##    100        0.2689             nan     0.1000    0.0040
    ##    120        0.2258             nan     0.1000    0.0015
    ##    140        0.1903             nan     0.1000    0.0025
    ##    150        0.1759             nan     0.1000    0.0009
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1281
    ##      2        1.5241             nan     0.1000    0.0890
    ##      3        1.4665             nan     0.1000    0.0674
    ##      4        1.4222             nan     0.1000    0.0553
    ##      5        1.3876             nan     0.1000    0.0504
    ##      6        1.3543             nan     0.1000    0.0428
    ##      7        1.3268             nan     0.1000    0.0331
    ##      8        1.3051             nan     0.1000    0.0359
    ##      9        1.2817             nan     0.1000    0.0296
    ##     10        1.2626             nan     0.1000    0.0331
    ##     20        1.1016             nan     0.1000    0.0150
    ##     40        0.9288             nan     0.1000    0.0084
    ##     60        0.8199             nan     0.1000    0.0065
    ##     80        0.7378             nan     0.1000    0.0050
    ##    100        0.6744             nan     0.1000    0.0040
    ##    120        0.6228             nan     0.1000    0.0027
    ##    140        0.5776             nan     0.1000    0.0029
    ##    150        0.5575             nan     0.1000    0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1874
    ##      2        1.4899             nan     0.1000    0.1271
    ##      3        1.4062             nan     0.1000    0.1030
    ##      4        1.3406             nan     0.1000    0.0846
    ##      5        1.2858             nan     0.1000    0.0732
    ##      6        1.2389             nan     0.1000    0.0619
    ##      7        1.1994             nan     0.1000    0.0558
    ##      8        1.1631             nan     0.1000    0.0545
    ##      9        1.1293             nan     0.1000    0.0455
    ##     10        1.1001             nan     0.1000    0.0493
    ##     20        0.8951             nan     0.1000    0.0199
    ##     40        0.6816             nan     0.1000    0.0106
    ##     60        0.5553             nan     0.1000    0.0088
    ##     80        0.4650             nan     0.1000    0.0050
    ##    100        0.3967             nan     0.1000    0.0022
    ##    120        0.3440             nan     0.1000    0.0026
    ##    140        0.2996             nan     0.1000    0.0017
    ##    150        0.2821             nan     0.1000    0.0021
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2322
    ##      2        1.4607             nan     0.1000    0.1626
    ##      3        1.3573             nan     0.1000    0.1324
    ##      4        1.2743             nan     0.1000    0.1037
    ##      5        1.2092             nan     0.1000    0.0893
    ##      6        1.1531             nan     0.1000    0.0776
    ##      7        1.1036             nan     0.1000    0.0640
    ##      8        1.0634             nan     0.1000    0.0650
    ##      9        1.0222             nan     0.1000    0.0537
    ##     10        0.9874             nan     0.1000    0.0482
    ##     20        0.7514             nan     0.1000    0.0267
    ##     40        0.5212             nan     0.1000    0.0099
    ##     60        0.4016             nan     0.1000    0.0107
    ##     80        0.3187             nan     0.1000    0.0049
    ##    100        0.2603             nan     0.1000    0.0023
    ##    120        0.2188             nan     0.1000    0.0019
    ##    140        0.1847             nan     0.1000    0.0015
    ##    150        0.1695             nan     0.1000    0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2380
    ##      2        1.4577             nan     0.1000    0.1548
    ##      3        1.3574             nan     0.1000    0.1292
    ##      4        1.2763             nan     0.1000    0.1009
    ##      5        1.2122             nan     0.1000    0.0832
    ##      6        1.1595             nan     0.1000    0.0822
    ##      7        1.1067             nan     0.1000    0.0807
    ##      8        1.0577             nan     0.1000    0.0566
    ##      9        1.0214             nan     0.1000    0.0628
    ##     10        0.9823             nan     0.1000    0.0474
    ##     20        0.7565             nan     0.1000    0.0244
    ##     40        0.5335             nan     0.1000    0.0112
    ##     60        0.4119             nan     0.1000    0.0041
    ##     80        0.3257             nan     0.1000    0.0048
    ##    100        0.2674             nan     0.1000    0.0049
    ##    120        0.2234             nan     0.1000    0.0020
    ##    140        0.1901             nan     0.1000    0.0017
    ##    150        0.1767             nan     0.1000    0.0025

Confusion matrix and accuracy

``` r
pred_gbm <-
  predict(
    model_gbm
    , validation
  )

confusionMatrix(
  pred_gbm
  , as.factor(validation$classe)
)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1654   29    0    1    3
    ##          B   15 1075   32    3   15
    ##          C    0   33  976   27   10
    ##          D    3    0   15  929   18
    ##          E    2    2    3    4 1036
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9635          
    ##                  95% CI : (0.9584, 0.9681)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9538          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9881   0.9438   0.9513   0.9637   0.9575
    ## Specificity            0.9922   0.9863   0.9856   0.9927   0.9977
    ## Pos Pred Value         0.9804   0.9430   0.9331   0.9627   0.9895
    ## Neg Pred Value         0.9952   0.9865   0.9897   0.9929   0.9905
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2811   0.1827   0.1658   0.1579   0.1760
    ## Detection Prevalence   0.2867   0.1937   0.1777   0.1640   0.1779
    ## Balanced Accuracy      0.9901   0.9651   0.9684   0.9782   0.9776

Gradient Boosted Trees Accuracy = 96.2%

Let’s check how well cross-validation performed and how it affected
model’s accuracy:

``` r
plot(
  model_gbm
)
```

![](Course-Project-F_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

### Model selection

Based on accuracy, the best model for our prediction is the Random
Forest model with 99% accuracy.

## Predicting classe on test set with selected model

Cleaning test data:

``` r
testing_clean <-
  testing_set[, -empty_cols]

testing_clean <-
  testing_clean[, -c(1:7)]

testing_clean %>%
  dim()
```

    ## [1] 20 53

Predicting on test set:

``` r
pred_rf_test <-
  predict(
    model_rf
    , testing_clean
  )

pred_rf_test
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
