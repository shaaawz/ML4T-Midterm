
# ML4T Midterm Study Guide
Topics in ML4T to study for Midterm

## NOTES FROM LECTURES:  
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

 
# Portfolio Statistics

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
