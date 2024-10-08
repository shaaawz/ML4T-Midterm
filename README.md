
# ML4T Midterm Study Guide
Topics in ML4T to study for Midterm - feel free to add your notes!

# NOTES FROM LECTURES:  
- [ ] Detailed Breakdown of Each Lesson: 
  - [Octavian Blaga Class Notes Spring 2017 ](https://docs.google.com/document/d/1BpDrMJDqx3sGt5-hoSTF3hJZVOf04hO_8ERdLHWP5A0/edit#heading=h.m1fuz8pxxckd)

## High Level Review Points from Lessons
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
## Company Worth 
  - [ ] Instrinsic value - Future Value/ Discount Rate
  - [ ] Book Value - Total assets - Intangible assets + Liabilities
  - [ ] Market Capitilization - # of shares * price of shares
  - [ ] Information can affect stocks:
      - Company specific
      - section specific
      - market wide
  - [ ] Capital Assets Pricing Model (CAPM)
      - Beta * market stock + alpha of stock on that day (residual)
      - Alpha's expected value is zero and random
      - Beta slope of the line of index's return compared to market returns
        - Larger beta better in upwards markets so you can ride the surge
        - Smaller beta better in downward markets, so you dont crash too much   
      - buy Index and hold
  - [ ] Active Managers
      - alpha can be predicted
      - pick individual stocks with different weights
  ## Technical Analysis 
  - [ ] Fundemental Analysis looks at value, earnings, dividends, cash flow, book value
  - [ ] Technical Analysis looks at:
      - patterns, trends in stock life
      - historic price and volume ONLY
      - Computes statistics on time series data called **indicators**
      - indicators are Heuristics (hints at buy/sell opportunities)
   - [ ] Technical Analysis is better thought of as a trading approach, rather than an investing approach
   - [ ] Why it might work:
      - There is important information in price changes - it reflects buyer/seller sentiments
      - Heuristics can work
   - [ ] When is Technical Analysis Effective?
      - Individual indicators are weak, but combining indicators has value, and can provide strong predictive power
      - Look for contrasts in data (compare stock to the market)
      - Most effective in shorter time periods
   - [ ] Fundemental Analysis most valuable over years (long periods of time)
   - [ ] **Technical Indicators:**
      - **Momentum** - over n days, how much has price changed?
        - Positive (buy) vs Negative Momentum, Steep vs Shallow Momentum
        - Momentum[t] = price[t]/price[t-n] - 1 where n = # of days
        - expected range: (-.5, .5)  
      - **Simple Moving Average** - average price of values in n day window
        - SMA = price[t] / price[t - n:t] - 1
        - expected range: (-.5, .5)  
        - Lagged Movement when graphed compared to actual price changes
        - Volatility affects how SMA is used (standard deviation tells us about volatility)
          - low volatility = smaller SMA used as trading trigger
          - high volatility = larger SMA used as trading trigger   
        - current price crosses SMA line indicates potential movement signal
        - Proxy fo underlying "true value" of company
        - string diverson can represent buy/sell opportunities
        - SMA + Momentum together creates a strong trading signal
      - **Bollinger Bands**
        - Look for cross from outside to inside is a trading signal because we expect it to go bakc to average
        - BollingerBand[t] = price[t] - SMA[t] / 2 * std[t]
        - expected range: (-1, 1)
   - [ ] **Normalization:** takes all factors and makes ranges (-1, 1) so they have same weighted effect
       - Normed = values - mean / values.std()

# NOTES FROM READINGS:  

## Probablistic Machine Learning (Chap 1)
- [ ] Machine Learning: A computer program is said to learn from experience E with respect to some class of tasks T,
and performance measure P, if its performance at tasks in T, as measured by P, improves with
experience E
- [ ] Probabilistic Perspective: all future values of quantity of interest are random varibles endowed with probablity distribution which describe a weighted set of possible values the variables may have
  - This is hte optimal approach to decision making under uncertainty
  - probabilistic modeling is used in other areas of science and engineering and provides a unifying framewokr w thse fields
- [ ] Supervised Learning: learn a mapping of f from inputs x to outputs y. x are features/covariates/predictors, y are the target,label, response. N (sample size) of the x-y pairs are given as a training set. Performance depends on the type of output we are predicting. In supervised learning, we assume that each input example x in the training set has an associated
set of output targets y, and our goal is to learn the input-output mapping
  - Goal of Supervised Learning is to automatically come up with classification models so as to reliably predict the labels for any given input
- [ ] Classification problems: output is unordered and mutally exclusive labels known as classes. Predicting the class label, given an input, is called pattern recognition. When there are two classes, it is called binary classification
- [ ] Function that is useful for challenging classification problems: Convolution Neural Networks (CNN)
- [ ] When we have small datasets of features, it is common to store them in an N × D matrix, in which each row represents an example, and each column represents a feature. This is known as a **design matrix**
- [ ] Tabular Data - when inputs are of variable size (sequences of words or social networks) rather than fixed-length vectors
- [ ] Featurization = process of converting tabular data into a fixed-size feature representation thus allowing a design matrix to be used for future processing, example: "bag of words" representation (discussed on page 24)
- [ ] Exploratory data analysis: before using ML, seeing if there are any obvious patterns which might giv eus hints on which method to choose or any obivous problems with the data (example:: label noise or outliers)
- [ ] Pair Plot - Scatterplot of variables, used for tabular dtaa with small number of features to see commonalities
- [ ] Dimensionality reduction - used for higher-dimension data, allows data to be visualized in 2d or 3d
- [ ] Decision rule = a rule that allows a class (target) to be distinguishable from other classes
- [ ] Decision boundaries - points that seperate one class from another class
- [ ] Decision Tree - Nested Decision rules and Decision Boundaries that use internal nodes that store feature indezes and corresponding threshold values to seperate data into classes
- [ ] Empirical Risk Minimization
- [ ] Misclassification Rate: performance of Supervised Learner to correctly classify features in the training set. Equation of page 6. I(e) is the binary indicator function, which returns 1 if e is true and returns 0 otherwise. Formula used if all errors are equal.
- [ ] An Asymmetric Loss Function is used if some errors are more costly than other errors.
- [ ] **Empirical risk** is the average loss of the predictor on the training set.
- [ ] Model Fitting / training: finidng a setting of the parameters that minimizes the emprical risk on the training set. This only minimizes in training set and does not necessilarly minimize expected loss on future data.
- [ ] In many cases, we will not be able to perfectly predict the exact output given the input, due to lack of knowledge of the input-output mapping (this is called **epistemic uncertainty or mode  uncertainty**), and/or due to intrinsic (irreducible) stochasticity in the mapping (this is called **aleatoric uncertainty or data uncertainty**).
- [ ] In statistics, the w parameters are usually called regression coefficients (and are typically denoted by β) and b is called the intercept. In ML, the parameters w are called the weights and b is called the bias. 
- [ ]  S good model (with low loss) is one that assigns a high probability to the true output y for each corresponding input x.
- [ ]  Regression - real valued quantity instead of class labels. Empircal risk when using quadratic loss is equal to Mean Squared Error or MSE
- [ ]  Simple Linear Regression model: f(x; θ) = b + wx. where w is the slope, b is the offset, and θ = (w, b) are all the parameters of the model. By adjusting θ, we can minimize the sum of squared errors until we find the **least squares solution**
- [ ]  Polynomial Regression Model of degree D can improve the fit of the data. This is called **feature preprocessing, also called feature engineering**
- [ ]   A model that perfectly fits the training data, but which is too complex, is said to suffer from **overfitting**
- [ ]   We create two subsets of the data: training and testing sets in order to calculate the different between the empirical risk of the trianing set and the "population risk" of the testing set. This difference is called the generalization gap. Large gap indicates overfitting
- [ ]   In practice, we need to partition the data into three sets, namely the training set, the test set and a validation set; the latter is used for model selection, and we just use the test set to estimate future performance (the population risk)
- [ ]   An arguably much more interesting task is to try to “make sense of” data, as opposed to just learning a mapping. That is, we just get observed “inputs” D = {x of n : n = 1 : N} without any corresponding “outputs” y of n. This is called **unsupervised learning**.
- [ ]   From a probabilistic perspective, we can view the task of unsupervised learning as fitting an unconditional model of the form p(x), which can generate new data x, whereas supervised learning involves fitting a conditional model, p(y|x), which specifies (a distribution over) outputs given inputs
- [ ]   **Unsupervised Learning**:
  - avoids the need to collect large labeled datasets for training, which can
often be time consuming and expensive
  - avoids the need to learn how to partition the world into often arbitrary categories. Tasks can be difficult to define and therefore not reasonable to expect machines to learn mappings
  - forces the model to “explain” the high-dimensional inputs, rather than just the low-dimensional outputs
- [ ] Unsupervised learning tries to find **clusters** in the data. The goal is to partition the input into regions that contain “similar” points.
- [ ] Latent factors of variations - factors that are hidden or unobsereved in low dimensions, but cause causation to the target
- [ ] **Self-supervised learning** - a sub approach to unsupervised leanring where we create proxy supervised tasks from unlabeled data. For example, we might try to learn to predict a color image from a grayscale image, or to mask out words in a sentence and then try to predict them given the surrounding context. The hope is that the resulting predictor xˆ1 = f(x2; θ), where x2 is the observed input and xˆ1 is the predicted output, will learn useful features from the data, that can then be used in standard, downstream supervised tasks.
- [ ] Unsupervised Learning is hard to evaluate becauset here is no ground truth to compare to
  - Common method to evaluate:  measure the probability assigned by the model to unseen test examples. We can do this by computing the (unconditional) negative log likelihood of the data - equation on page 16
- [ ] This treats the problem of unsupervised learning as one of **density estimation**. The idea is that a good model will not be “surprised” by actual data samples
- [ ] **Reinforcement learning** -  In this class of problems, the system or agent has to learn how to interact with its environment. This can be encoded by means of a policy a = π(x), which specifies which action to take in response to each possible input x (derived from the environment state).
- [ ] The difference from supervised learning (SL) is that the system is not told which action is the best one to take (i.e., which output to produce for a given input). Instead, the system just receives an occasional reward (or punishment) signal in response to the actions that it takes. This is like learning with a critic, who gives an occasional thumbs up or thumbs down, as opposed to learning with a teacher, who tells you what to do at each step.


## Handbook of AI and Big Data (Chap 1, 2, 7)

### On Machine Learning Applications in Investments
- [ ] Motivations
    - Enhanced Decision-Making: ML can analyze vast amounts of data quickly, enabling more informed investment decisions.
    - Predictive Analytics: Algorithms can identify patterns and trends, helping to forecast market movements and asset performance.
    - Risk Management: ML models can assess and manage risk more effectively by analyzing historical data and identifying potential pitfalls.
    - Cost Efficiency: Automation through ML can reduce operational costs and improve efficiency in trading and portfolio management.
    - Personalization: ML can tailor investment strategies to individual preferences and risk tolerances, enhancing client satisfaction.
- [ ] Challenges
    - Data Quality and Quantity: Accessing high-quality, relevant data can be difficult. Inaccurate or biased data can lead to poor model performance.
    - Model Overfitting: Complex models may perform well on historical data but fail to generalize to new, unseen data.
Regulatory and Ethical Issues: Compliance with regulations and addressing ethical concerns, such as algorithmic bias, are significant challenges.
    - Market Dynamics: Financial markets are influenced by numerous unpredictable factors, making it difficult for models to adapt quickly.
    - Interpretability: Many ML models are seen as "black boxes," making it hard for investors to understand and trust their outputs.
- [ ] Solutions
    - Robust Data Management: Implementing strong data governance practices to ensure data quality and relevance.
    - Regular Model Validation: Continuously testing and updating models with new data to prevent overfitting and improve adaptability.
    - Transparent Algorithms: Developing interpretable models and tools that help explain how decisions are made can build trust with stakeholders.
    - Hybrid Approaches: Combining ML with traditional financial analysis methods can enhance robustness and accuracy.
    - Staying Informed: Keeping up with regulatory changes and incorporating compliance checks into ML processes can mitigate legal risks.
- [ ] ML techniques can deliver performance above and beyond traditional approaches if applied to the right problem.
- [ ] The source of ML algorithms’ outperformance includes the ability to consider nonlinear and interaction effects among the input features.
- [ ] Ensembling of ML algorithms often delivers better performance than what individual ML algorithms can achieve.

### ALTERNATIVE DATA AND AI IN INVESTMENT  RESEARCH
- [ ] Alternative Data and AI in Investment Research refers to the use of non-traditional data sources and advanced analytics to enhance investment decision-making. Here’s a breakdown of both components:
- [ ] Alternative Data Definition: Alternative data includes any data that is not commonly used in traditional financial analysis. This can encompass a wide range of information, such as:
    - Social media sentiment
    - Web traffic and search trends
    - Satellite imagery (e.g., for tracking retail foot traffic)
    - Supply chain data
    - Credit card transactions
    - Weather data
- [ ] Benefits:
    - Timeliness: Alternative data can provide real-time insights that traditional data sources may lag behind.
    - Unique Insights: It can reveal trends and consumer behaviors that are not captured in standard financial reports.
    - Competitive Advantage: Firms that effectively utilize alternative data can gain an edge over competitors.
    - The impact of AI and alternative data on investment research is seen as evolutionary rather than revolutionary, according to - [ ] Goldman Sachs. They view nonstructured, alternative, and big data as integral components of their research process, alongside analytical tools like AI, machine learning (ML), and natural language processing (NLP). Goldman Sachs does not create strict distinctions between traditional and alternative data; instead, they integrate various data types and analyses as needed, combining unstructured data with traditional metrics.
- [ ] Their approach emphasizes collaboration between subject matter experts and data scientists to minimize biases in algorithms and data. As the field evolves, they anticipate that the integration of AI and alternative data will become so seamless that terms like "big" and "alternative" may eventually lose significance in the research process. This integrated, iterative approach is expected to enhance their investment research outcomes over time.

### MACHINE LEARNING AND BIG DATA TRADE EXECUTION SUPPORT
- [ ] Feature Importance refers to a technique used in machine learning to determine the relevance of different input variables (or features) in predicting an outcome. By calculating a score or rank for each feature, it helps to identify which variables significantly influence the model's predictions. The main goal is to reduce the complexity of models by highlighting the most important inputs. This allows traders to focus on the most meaningful factors when selecting trading strategies. Done with Permutation Importance and Tree-based Methods
- [ ]  Transaction Cost Analysis (TCA): is a comprehensive approach used by traders and investment firms to evaluate the costs associated with executing trades. It helps assess the efficiency and effectiveness of trading strategies by analyzing various costs involved in the transaction process.
- [ ] Steps ML models should implement:
    - Cleansing and Normalizing Data
    - Chucking / Breaking Down Data into Subsets for Easier Interpretation
    - Data labeling and Testing and Training Datasets
    - Selecting the Model's Features
    - Selecting the ML Model's Library and Training a supervised Model
    - Selecting a Feature Importance Approach
- [ ] Semi-Supervised Learning is a machine learning approach that combines both labeled and unlabeled data to improve the learning process. It lies between supervised learning, which uses only labeled data, and unsupervised learning, which uses only unlabeled data.
- [ ] natural language processing (NLP):  a subfield of artificial intelligence focused on the interaction between computers and human language. It involves the development of algorithms and models that enable machines to understand, interpret, and generate human language in a meaningful way.
    
- [ ] A stock clustering model is a machine learning approach used to group stocks into clusters based on their characteristics, behaviors, or performance metrics. The goal is to identify similarities among stocks, which can help investors and analysts make more informed decisions about portfolio management, risk assessment, and trading strategies.

## Deep Learning (Chap 1, 2.1, 2.2)
- [ ] Bayes Theorem: is a fundamental concept in probability theory that describes how to update the probability of a hypothesis based on new evidence. It provides a mathematical framework for revising beliefs in light of new data.
- [ ] Prior Probability: P(H) - The initial probability of a hypothesis before observing any new evidence.
- [ ] Likelihood P(E|H) - The probability of observing the evidence given that the hypothesis is true.
- [ ] Marginal Probability P(E): The total probability of observing the evidence under all possible hypotheses.
- [ ] Posterior Probability P(H|E): The updated probability of the hypothesis after taking into account the new evidence.
- [ ] Formula: P(H|E) = P(E|H) * P(H) / P(E)

## Introduction to Statistical Learning with Applications in Python (Chap 1, 2.2, 3.1-3.3, 3.5, 8.1 - 8.2)
### Chapter 2: Statistical Learning
- [ ] input variables X1, X2, . . . , Xp.
  - Also known as predictors, independent variables, features 
- [ ] output variables Y
    - Also known as dependent variables, responses
- [ ] General Formula: Y = f(X) +  ε
    - Here f is some fixed but unknown function of X1, . . . , Xp, and  ε is a *random
error term*, which is independent of X and has mean zero. In this formula, f represents the systematic information that X provides about Y
- [ ] Reducible Error vs Irreducible Error (page 17)
    - The accuracy of Yˆ as a prediction for Y depends on two quantities, which we will call the reducible error and the irreducible error. In general, fˆ will not be a perfect estimate for f, and this inaccuracy will introduce some error. This error is reducible because we can potentially improve the accuracy of fˆby using the most appropriate statistical learning technique to estimate f. However, even if it were possible to form a perfect estimate for f, so that our estimated response took the form Yˆ = f(X), our prediction would still have some error in it! This is because Y is also a function of ε, which, by definition, cannot be predicted using X. Therefore, variability associated with ε also affects the accuracy of our predictions. This is known as the irreducible error, because no matter how well we estimate f, we cannot reduce the error introduced by ε.
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

### Chapter 3: Linear Regression (3.1 - 3.3)
- [ ] Simple linear regression
  - lives up to its name: it is a very straightforward
simple linear approach for predicting a quantitative response Y on the basis of a single predictor variable X. It assumes that there is approximately a linear
relationship between X and Y.
  - Y ≈ β0 + β1X
  - β0 and β1 are two unknown constants that represent the intercept and slope terms in the linear model. Together, β0 and β1 are known as the model coefficients or parameters.
- [ ] RSS (Residual Sum of Squares)
  - measures the level of variance in the error term, or residuals, of a regression model. The smaller the residual sum of squares, the better your model fits your data; the greater the residual sum of squares, the poorer your model fits your data
- [ ]  Population Regression Line
  - which is the best linear approximation to the **true** relationship between X and Y
- [ ] Least Square Line
  -  it is the least squares **estimate** for f(X) based on the observed data
- [ ] P-value
  - a small p-value indicates that it is unlikely to observe such a substantial association between the predictor and the response due to chance, in the absence of any real association between the predictor and the response
- [ ] Residual Standard Error (RSE) (formula on page 78) and used in calculating R2
  -  an estimate of the standard deviation of ε. Roughly speaking, it is the average amount that the response will deviate from the true regression line.
  -  The RSE provides an absolute measure of lack of fit of the model to the data
- [ ] Correlation
  - another measure of the linear relationship between X and Y

### Chapter 8: Tree Based Models (8.1 - 8.2)
- [ ]  Predicted Values are known as terminal nodes or leaves of the tree. Decision trees are typically drawn upside down, with leaf nodes at the bottom of the tree. The points along the tree where the predictor space is split are referred to as internal nodes.
- [ ]   it is computationally infeasible to consider every possible partition of the feature space into J boxes. For this reason, we take a top-down, greedy approach that is known as recursive binary splitting. The approach is top-down because it begins at the top of the tree (at which point all observations belong to a single region) and then successively splits the predictor space; each split is indicated via two new branches further down on the tree. It is greedy because at each step of the tree-building process, the best split is made at that particular step, rather than looking ahead
and picking a split that will lead to a better tree in some future step.
- [ ]  Best to grow a very large tree and then **prune** the tree back in order to obtain a subtree. Uur goal is to select a subtree that leads to the lowest test error rate. Given a subtree, we can estimate its test error using cross-validation or the validation set approach.
- [ ]  *Cost complexity pruning*—also known as weakest link pruning - explained on page 336
- [ ]  **Building A Regression Tree**
  - 1. Use recursive binary splitting to grow a large tree on the training data, stopping only when each terminal node has fewer than some minimum number of observations.
  - 2. Apply cost complexity pruning to the large tree in order to obtain a sequence of best subtrees, as a function of α.
  - 3. Use K-fold cross-validation to choose α. That is, divide the training observations into K folds. For each k = 1, . . . , K:
    - (a) Repeat Steps 1 and 2 on all but the kth fold of the training data.
    - (b) Evaluate the mean squared prediction error on the data in the left-out kth fold, as a function of α.
    - Average the results for each value of α, and pick α to minimize the average error.
  - 4. Return the subtree from Step 2 that corresponds to the chosen value of α
- [ ]  A **classification tree** is very similar to a regression tree, except that it is used to predict a qualitative response rather than a quantitative one. For a classification tree, we predict that each observation belongs to the most commonly occurring class of training observations in the region to which it belongs.
- [ ]   If the relationship between the features and the response is well approximated by a linear model as in (8.8), then an approach such as linear regression will likely work well, and will outperform a method such as a regression tree
that does not exploit this linear structure. If instead there is a highly nonlinear and complex relationship between the features and the response as indicated by model (8.9), then decision trees may outperform classical approaches. 


## What Hedge Funds Really Do (Chap 2, 4, 5, 7, 8, and 12)
###Chapter 2: So You Want to Be a Hedge Fund Manager
- [ ]  4 Strageties of Funds
  - Equity - emphasis on stock selection
  - Arbitrage - where managers seek instances where price relationships between assets fall outside normal variation, and bet on the relationship returning to normal
  - Momentum/Direction - where managers have a macro view of hte probable direction of the price in a market
  - Event-Driven - trades instigate based on an event (war, merger, supply distruption)
- [ ]  Long-only
  - make money only if the asset rises in price
- [ ]  Short-only
  - profit if the asset price falls
- [ ]  Hedged
  - long and short
- [ ]  "130/30" equity strategy: 130 long (borrowing 30% over and above 100% equity capital) and 30% short of the portfolio
- [ ] CAGR (compounded annual growth return)
  - the rate of return that an investment would need to have every year in order to grow from its beginning balance to its ending balance, over a given time interval.
  - if CAGR = 7.2% - that means that a portfolio that grows 7.2% on average each year, will double in size in 10 years
  - Rule of 72 (approximation of compounding)
    - you can approximate the number of periods that will be needed for a sum to double by dividing the CAGR (in whole numbers) into the number 72
    - 6% CAGR: 72/6 = 12 years to double
    - if portfolio doubled over 15 years, that means CAGR = (72/15) = 4.8%


### Chapter 4: Market-Making Mechanics
- [ ] Markets - locations where sellers and customers convene to exchange goods, or financial instruments
- [ ] Investors transact exchanges thorugh brokerage firms, who in turn trade with market makers (specialists)
- [ ] NASDAQ - no floor, all tradining occurs electronically, no specialists
- [ ] Market Spreads
  - Bid Price: the price at which a buyer may purchase a stock
  - Ask Price: The price at which the seller will sell the stock
  - Market Spreads: The difference between the ask and bid price. It represents the profit opportunity that induces the specialists to connect sellers and buyers
  - Highly liquid markets tend to have smaller spreads 
- [ ] Types of Orders
  - Buy, Sell, "At market", "Limit order"
  - Selling Short - bet that a stock's price will fall
    - Investor borrows shares from a holder and sells them earning the proceeds of the sale. Later, they buy the same number of shares and return them to the lender
    - Investor pays interest to the owner of the shares borrowed and the dividents entitled during borrowing period
    - Two transactions involved: sell to open and buy to close
    - risky because long-term trends move upwards so the premise of a short is the stock will move in the opposite direction
    - few hedge funds are short only
  - Going Long - stock price will rise
  - Stop orders -place a market order if the price of the stock falls more than a certain threshhold (stop loss)
  - Trailing stops - equivalent to stop orders, but make the condition the most recent high to preserve the most of hte gains
- [ ] The Order Book
  - Shares being offered (ask) (WANTING TO SELL)
  - Shares being requested (bid) (WANTING TO BUY)
  - More shares being offered than requested suggests share price will decline
- [ ] Dark pools - informal exchanges among brokerage firms pools
- [ ] Competitive advantage hinges on milliseconds
- [ ] Front running - broker issues trades in advance of those of its clients, knowing the price movements that will result in executing the client orders (ethics are dubious)

### Chapter 5: Introduction to Company Valuation
- [ ] Current price below estimated value - opportunity to go long
- [ ] If asset is overpriced, it presents a short opportunity
- [ ] **Margin of safety**: only buy stocks whose price was well below investor's estimate of true value
- [ ] **Fundamental Analysis** based on business operations & finances
- [ ] **Technical Analysis** predicts stock prices based on past price behaviors, also called "charting"
- [ ] 3 Methods for Estimating Company Value
    - Book Value: sum of assets & liabilities
    - Instrinsic Value: Future dividends to be paid by the company
    - Earnings Growth: Projection of expanded earnings into the future
- [ ] **Book Value**
    - doesn't reflect future prospects
- [ ]  Based on Assets - productive items that the business owns
    - Tangible Assets: Quickly can be converted to cash. examples: factories, cars, buildings
    - Intangible Assets: Cannot be quickly converted to cash. examples: patents
- [ ] Liabilities: financial obligations, debts, contracts, commitments into the future 
- [ ] Net Worth: approximation of how much money is left after company sells all assets and pays off all liabilities
- [ ] Book value does not equal market capitalization (value appraised by market value)
- [ ] Book Value is usually below actual price firm is worth, except in a recession, then book value may overstate true value
- [ ] Price to Book Ratio = share price / book value per price
- [ ] **Instrinsic Value** Dividend-based valuation
- [ ] Takes into account time value of money
- [ ] Assets more valuable if they generate cash into future
- [ ] **Discounting**: negotiating a price that converts a stream of future cash flows into lump sum
- [ ] Present Value = future value / (1 + Discount Rate)^N
- [ ] N = number of years until payment
- [ ] Discount Rate (DR): rate of return the investor could receive from investing in a best alternative asset
- [ ] Dividends - stream of payment paid regular basis for infinite future
- [ ] Instrinsic Value: present value of all future returns, challenging to estimate future value cash flow
- [ ] The Dividend Discount Model: $1 loses value over time
- [ ] For Long-Term ownership of an asset: PV = FV / DR

### Chapter 7: Capital Assets Pricing Model
- [ ] Core Belief: very few investors can produce sustained returns superior to market averages
- [ ] Core of CAPM: Distinguishing between stock returns that derive from broad market movements and those that do not
- [ ] Positive Correlation between stock returns - events affect both assets in the same general way
- [ ] Returns = price[t] / price[t-1] -1
- [ ] Basic Measure of Relationship between stocks: Correlation
- [ ] Correlation Coefficient measures frequency in which prices of two assets mov ein the same direction
- [ ] -1 = no correlation, 0 = no visible relationship, 1 = perfect positive relationship
- [ ] Linear relationship: Return = Beta * SPX + Alpha
- [ ] Alpha: systematic difference in performance of a stock over and above the market, presumed to be zero
- [ ] Beta: stock's price volatility relative to the overall market. Beta determines how much more/less a stock will stock will change with regards to market
- [ ] Correlation Coefficients: capture "tightness" of the scatter around regression line, summarizes pattern in that scatter
- [ ] higher correlation coefficient suggests that effcts to market will have similar effects to stock
- [ ] "Buying Beta": Investing in stock more volatile than market
- [ ] "Buying Alpha": Finding stocks that systematically outperform market
- [ ] Return of Portfolio = w1 * R1 + w2 * R2 +....wN * RN
- [ ] where R = Beta of stock * market + Alpha of stock
- [ ]  -w is used when shorting a stock
- [ ]  Beta Balanced Portfolio: sum(beta_i * wi) = 0 and sum(|w_i|) = 1.0

### Chapter 8: Efficient Market Hypothesis
- [ ] Most markets are efficient - information that can affect prices travel quickly throughout a market and prices are affected accordingly
- [ ] Measure of efficiency: speed with which stock's price adjusts to company relevent information
- [ ] Investor self interest causes information to reflect in stock prices quickly
- [ ] High Tradining Volume Stocks w Transparency and wide disclosure of relevent info have high efficiency
- [ ] Niche Markets (Poor info transmission) are less efficient
- [ ] 3 Versions of EMH:
    - Weak Form: future asset prices cannot be predicted using historical price/volume data
    - Semi-Strong Form: asset prices adjust immediately to all publicly available information
    - Strong Form: Asset prices adjust immediately to reflect all relevent info, including available to insiders

### Chapter 12: Overcoming Data Quirks to Design Trading Strategies
- [ ] Since stock price data is widely avaiable and must be used with special care, there are pitfalls that one must avoid in price data and coping strategies to overcome these pitfalls
- [ ] Actual vs Adjusted Stock Price Data - stock prices change for reasons other than market supply and demand.
- [ ] Adjusted Price accounts for splits and dividends issued
- [ ] Split the stock: divide each old share into a larger number of new shares in order to keep stock accessible for retail groups of investors (value of company doesn't change, but each share is now less expensive and old shares are worth N new shares)
- [ ] Reverse Splits also exist
- [ ] Dividends: Shareholders can earn income from stocks without selling them if the company board of directors declares dividends a payout to shareholds of cash that is a portion of hte company's annual earnings
- [ ] Payout ratio
- [ ] Value of stock needs to take into account dividend payouts. Dividend Yield: amount of annual dividend per share / share price
- [ ] Breaks in Series and Missing Data: Necessary to fill the data with reasonable guesses
- [ ] fill forward -treat missing values as the same level as the last known value
- [ ] fill backwareds - missing values at the beginning of the series
- [ ] fill forward first, fill backwards only where you can't fill forward
- [ ] Missing / Delisted stocks dude to when company goes public or company being acquired by another, going private, or out of business
- [ ] Only 30% of hedge funds that existed 10 years ago are still in business today
- [ ] "trend to omit" - analysts scrubbing sample to include only companies that operate during a specific period omits more failures than successes, causing survivor bias
- [ ] Analyst goal should be to sanity check the data sample and ensure that it will give a realisitic and relevant view of the data - rather than an idealistic and uncritical view. 
# ADDITIONAL RESOURCES: 

- [ ] https://lucylabs.gatech.edu/ml4t/#

- [ ] https://github.com/manikandan-ravikiran/ML4T-Notes/tree/master
