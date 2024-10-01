
# ML4T Midterm Study Guide
Topics in ML4T to study for Midterm

# NOTES FROM LECTURES:  

## Comparing Plots

- [ ] Expected Value 

  - Mean of random value in independent repetitions of the experiment 

- [ ] PMF: Probability Mass Function 

- [ ] Kurtosis 

  - Tails of distribution, positive kurtosis = fat tails (more occurrences outside in the tails than normal distribution) 

- [ ] Histograms for comparing daily returns  

  - Lower/higher returns vs lower/higher volatility  

- [ ] Daily Returns 

- [ ] Scatterplots 

  - Fit Line – Linear Regression 

- [ ] Alpha – Where line intercepts vertical axis. 

  - If Alpha is positive, stock performs better than S&P each day 

- [ ] Beta – Slope (greater beta means stock is more reactive to market). If slope=2 means when market goes up 1%, stock goes up 2% 

  - Slope DOES NOT equal correlation  

- [ ] Correlation is measure of how tightly points fit the line) 

  - (0 – 1) Zero means not correlated, 1 means highly correlated 

  - Find correlation: df.corr(method = pearson) 

  - In statistics, the Pearson correlation coefficient (PCC) is a correlation coefficient that measures linear correlation between two sets of data. 

 
## Portfolio Statistics

- [ ] Bollinger Bands - Bollinger Bands are a technical analysis tool used by traders to assess price volatility and identify potential entry and exit points in the stock market. They are made up of three lines:  

  - Middle band: A simple moving average  
  
  - Upper band: Set above the moving average by a certain number of standard deviations of price  
  
  - Lower band: Set below the moving average by a certain number of standard deviations of price 
 

- [ ] Normalize: prices/prices[0] 

- [ ] Cumulative Returns: the total gain or loss of an investment over a specific period of time.   

- [ ] Allocations – amount distributed to each stock in a portfolio 

- [ ] Position Value – reflects how much a stock was worth each day 

- [ ] Portfolio Value – daily total value of the portfolio 

- [ ] Daily Returns – how much price goes up or down (prices of today minus price of yesterday) - 1 

- [ ] Average Daily Returns (mean of all daily returns) 

- [ ] Standard deviation of Daily Returns (risk) 

- [ ] Sharpe Ratio – Return in context of risk

## Assessing Learning Statistics
- [ ] Linear Regression
  - Parametric models can extrapolate where data will continue to go
- [ ] KNN Solution
  - Find k closest values, take mean of their y to get ŷ (y prediction)
  - querying model from left to right
  - model cannot extrapolate, ends stay stagnant
- [ ] Overfit - matches dataset exactly (k=1)
  - as we decrease k, we are more likely to overfit
  - as d (degrees of polynomial) increases, we are more likely to overfit
- [ ]  RMSE error (Root Mean Squared Error)
  - square error at each point, square it, and take the square root of that #
  - RMSE = sqrt [(Σ(Ytest – Ypredict)²) / n]
- [ ]  Cross Validation
  - resampling and sample splitting methods that use different portions of the data to test and train a model on different iterations
  - training (60%) test (40%)
  - Training Data is always before testing data (time sensitive)
- [ ]  As RMS increases, correltaion (mostly, but not always) decreases
- [ ]  Linear Regression VS KNN
  - Space Saving? Linear Regression better
  - Compute time to train? KNN better
  - Compute time to query? Linear Regression better
  - Ease to add new data? KNN better
- [ ]  Entropy
  - measures diversity and randomness of sample
- [ ]  Ensemble Learners
  - Train models with different leaf sizes/data sets/differing degrees
  - lower error, less overfitting
- [ ]  Bootstrap Aggregation - Bagging
  - random samples with replacement (same data can be used in multiple bags)
  - take in data, shuffle data, and insert n into bag
## Computational Investing
- [ ] Types of Funds
  - ETF: Electronically Exchange Traded Funds (usually 3-4 letters)
    - Buy/sell like-stocks or baskets of stocks
    - Transparent
    - Liquid - ease one can buy/sell
  - Mututal Funds - Exchange Through Broker (5 letters)
    - Buy/Sell at end of the day
    - Quarterly disclosure
    - Less transparent
  - Hedge Funds
    - Buy/Sell by agreement
    - No disclosure
    - Not transparent
    - More Risky, More Profits
- [ ] Large "Cap" / Capilization
  - How much is a company worth according to( # of shares that are outstanding * price of the stock)
- [ ] Assets Under Management (AUM)
  - How Managers are Compensated:
    - ETFs - Expense Ratio (AUM) - Range: 0.01% - 1.00%
    - Mutual Funds - Expense Ratio (AUM) - Range: 0.5% - 3.00%
    - Hedge Funds - "Two & Twenty" - 2.00% AUM + 20% of the profits
- [ ] How Funds Attract Investors
  - Who Are Investors?
    - Rich Individuals
    - Institutions
    - Funds of Funds (Groups of Individuals / Institutions)
  - Why Investors choose to invest?
    - Track record of Manager (returns in the past 5 years)
    - Simulation & Story (Description of Method used)
    - Good Portfolio Fit
  - [ ] Goals of Funds
    - Beat a benchmark (SP500)
    - Absolute return
  - [ ] Metrics used to assess:
     - Cumulative returns
     - Volatility (standard deviation)
     - Risk Adjusted Reward (Sharpe Ratio)


# NOTES FROM READINGS:  
## Introduction to Statistical Learning with Applications in Python
### Chapter 2: Statistical Learning
- [ ] input variables X1, X2, . . . , Xp.
  - Also known as predictors, independent variables, features 
- [ ] output variables Y
    - Also known as dependent variables, responses
- [ ] General Formula: Y = f(X) +  ε
      - Here f is some fixed but unknown function of X1, . . . , Xp, and  ε is a *random
error term*, which is independent of X and has mean zero. In this formula, f represents the systematic information that X provides about Y
- [ ] Reducible Error vs Irreducible Error (page 17)
    - The accuracy of Yˆ as a prediction for Y depends on two quantities, which we will call the reducible error and the irreducible error. In general, fˆ will not be a perfect estimate for f, and this inaccuracy will introduce some error. This error is reducible because we can potentially improve the accuracy of fˆby using the most appropriate statistical learning technique to estimate f. However, even if it were possible to form a perfect estimate for f, so that our estimated response took the form Yˆ = f(X), our prediction would still have some error in it! This is because Y is also a function of, which, by definition, cannot be predicted using X. Therefore, variability associated with " also affects the accuracy of our predictions. This is known as the irreducible error, because no matter how well we estimate f, we cannot reduce the error introduced by ε.
- [ ] Training Data - we will use these training observations to train, or teach, our method how to estimate f
- [ ] Parametric Model: It reduces the problem of estimating f down to one of estimating a set of parameters. Assuming a parametric form for f simplifies the problem of
estimating f because it is generally much easier to estimate a set of parameters, such as β0, β1, . . . , βp in the linear model 
  - Two step model based approach
  -  First, we make an assumption about the functional form, or shape, of f.  Once we have assumed that f is linear, the problem of estimating f is greatly simplified. Instead of having to estimate an entirely arbitrary p-dimensional function f(X), one only needs to estimate the p + 1 coefficients β0, β1, . . . , βp. Linear Model.
  -  After a model has been selected, we need a procedure that uses the training data to fit or train the model.
- [ ] Flexible Model - can fit many different possible functional forms for f
- [ ] Non-Parametric Model - methods do not make explicit assumptions about the functional form of f. Instead they seek an estimate of f that gets as close to the
data points as possible without being too rough or wiggly. Such approaches can have a major advantage over parametric approaches: by avoiding themassumption of a particular functional form for f, they have the potential to accurately fit a wider range of possible shapes for f.
- [ ] Thin-Plate Spline - This approach does not impose any pre-specified model of f. It attemps to produce an estimate for f that is as close as possible to the observed data, subject to the fit, and *smooth*
- [ ] Overfitting
  - a method that yields a smal training MSE but a large test MSE
  - Model follow the errors, or noise, too closely

- [ ] Supervised Statistical Learning Problems
  -  For each observation of the predictor measurement(s) xi, i = 1, . . . , n there is an associated response measurement yi. We wish to fit a model that relates the response to the predictors, with the aim of accurately predicting the response for future observations (prediction) or better understanding the relationship between the response and the predictors (inference).
- [ ] Unsupervised Statistical Learning Problems
  - describes the somewhat more challenging situation in which for every observation i = 1, . . . , n, we observe
a vector of measurements xi but no associated response yi. It is not possible to fit a linear regression model, since there is no response variable
to predict. In this setting, we are in some sense working blind; the situation is referred to as unsupervised because we lack a response variable that can supervise our analysis.
- [ ] Cluster Analysis / Clustering
  - The goal of cluster analysis is to ascertain, on the basis of x1, . . . , xn, whether the observations fall into analysis
relatively distinct groups.
- [ ] Regression Problems
  - Quantitative variables
- [ ] Classification Problems
  - Qualitative (categorical) variables
- [ ] Mean Squared Error
  - MSE = Σ(yi − pi)2n
  - measures how well predictions match observed data, small MSE means predictions close to responses
  -  trainings MSE vs test MSE
- [ ] Degrees of Freedom - a quantity that summarizes the flexibility of a curve
- [ ] Cross-Validation - is a crossmethod for estimating the test MSE using the training data.
- [ ] [Bias / Variance trade off](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
  - Challenge lies in finding a mathod for which variance and bias are low
  - low bias, high variance means curve passes thorugh every single training observation
  - low variance, high bias means fitting horizontal line to the data
  - If we are mainly interested in inference, then restrictive models are much more interpretable. For instance, when inference is the goal, the linear model may be a good choice since it will be quite easy to understand the relationship between Y and X1, X2, . . . , Xp.
  - In contrast, very flexible approaches can lead to such complicated estimates of f that it is difficult to understand how any individual predictor is associated with the response.
  - Finally, fully non-linear methods such as bagging, boosting, support vector machines with non-linear kernels, and neural networks (deep learning), discussed in
are highly flexible approaches that are harder to interpret.
- [ ]  Bayes Classifier
  - Computing the Bayes classifier is impossible. Therefore, the Bayes classifier serves as an unattainable gold standard
against which to compare other methods. 
- [ ]  K-Nearest Neighbors
  -  Given a positive integer K and a test observation x0, the KNN classifier first identifies the K points in the training data that are closest to x0, represented by N0. It then estimates the conditional probability for class j as the fraction of points in N0 whose response values equal j
  -  Better for unbiased
  -  finds average of the y-values of the K neighbors to x0 test observation

