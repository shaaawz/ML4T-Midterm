
# ML4T Midterm Study Guide
Topics in ML4T to study for Midterm

# NOTES FROM LECTURES:  
- [ ] Pandas Dataframes 

  - Slicing

- [ ] NumPy Arrays 

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
- [ ] Overfitting
  - a method that yields a smal training MSE but a large test MSE
  - Model follow the errors, or noise, too closely
- [ ] [Bias / Variance trade off](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
  - Challenge lies in finding a mathod for which variance and bias are low
  - low bias, high variance means curve passes thorugh every single training observation
  - low variance, high bias means fitting horizontal line to the data
- [ ]  Bayes Classifier
- [ ]  - [ ] Mean Squared Error
  - MSE = Σ(yi − pi)2n
  - measures how well predictions match observed data, small MSE means predictions close to responses
- [ ] 
