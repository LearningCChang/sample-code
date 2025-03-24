# sample-code
#league-of-legends-analysis
Sample Code: Analysis of League of Legends' 2022 dataset. 

Final Project for EECS 398 at U-M

# Contributions to view: data cleaning + baseline model + final model (decision tree classifier)

Website (with images): https://learningcchang.github.io/league-of-legends-analysis/

## Introduction
League of Legends is a popular multiplayer video game where two teams of 5 players battle to destroy each other’s base. The game map is split up into three lanes: top, middle, and bottom. Each team has towers in each lane that protect the base, and players of the opposing team have to destroy these turrets to advance and destroy the enemy base. The question we are looking to investigate further is how much does the probability of winning a match change based on key performance indicators such as team kills per minute, total gold, and the first to capture the midtower.
We chose to work with the League of Legends’ dataset because we felt that working on a dataset from a popular video game would lead to a more personalized project that would be not only more appealing on our resumes but also since it’s something we are somewhat familiar with. Expanding on this, our familiarity with video game mechanics and terminology allows us to understand the significance of the various features better. Furthermore, working on real-world data we are passionate about increases our motivation for this project, and our analysis of this dataset could uncover insights that are valuable not only for academic purposes but also for practical applications in our gaming experiences.

## Data Cleaning and Exploratory Data Analysis

### Data cleaning
The main data-cleaning step we took within our modeling was to drop all N/A values. We noticed within the CSV file for the League of Legends results that all of the N/A values for the columns we were looking at fell within individual statistics, which we were not interested in for the purpose of our model. By dropping those N/A values, we shrunk our data frame considerably and are now only looking at collective team results.

<iframe
  src="assets/gold_kpm_scatter.html"
  width="800"
  height="400"
  frameborder="0"
></iframe>

After cleaning our data by removing null values and selecting relevant features, we analyzed the relationship between team KPM (Kills Per Minute), total gold, and match outcomes.

The scatter plot above shows an interesting relationship between a team's KPM and their total gold accumulation, colored by match outcome. We can observe that teams with higher KPM and total gold tend to win more frequently, suggesting these are important predictors of match outcome.

### Univariate Analysis

<iframe
  src="assets/kpm_univariate.html"
  width="800"
  height="400"
  frameborder="0"
></iframe>

This histogram shows the distribution of team kills per minute (KPM) across all games. 
A key trend we observed is that most games have a KPM between 0.2 and 0.6, with the peak between 0.3 and 0.4.
The histogram is right-skewed, with some exceptionally aggressive games, where the kpm is greater than 1.0.
This metric is important for our analysis since it helps us undestand the pace of the game and team aggression levels during the game, which may correlate to whether or not teams can secure a victory in their game.

### Bivariate Analysis
<iframe
  src="assets/first-tower-winrate.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
This pie chart displays the percentage of games teams win and lose when they successfully destroy the first mid-tower of the game. 
Our data suggests that teams who destroy the mid-tower first have a higher win rate than their counterparts. 
However, we must keep in mind that many other factors contribute to a game’s win, such as team kills, gold, and players’ in-game decisions.

### Interesting Aggregates
An issue that we had with our initial data frame was that we had a TON of data to work with. 
This would lead to a major performance slowdown on our computers to create charts and decision boundaries. 
To combat this, we essentially binned our data into 90 “wins” and 90 “losses” to speed up the modeling time and minimize the impacts of outliers within the larger data frame. 
<br>
The process of doing that was to first make two new data frames, with one containing all losses and one containing all wins. 
Each data frame had exactly half the columns of the initial data frame. 
Then, we arranged the data in both new DataFrames by now making it 90 columns (since the number of columns of each DataFrame was 10616, which did not have many useful divisors, 
we just arbitrarily picked 90 each as a balance between an efficient set of data without overly cutting down our data into groups that were wholly unrepresentative of the current data). 
Because the division did have a remainder, one column had an irregular amount of aggregation. From there, we then grouped the data by the means of the data in each column, thus making the grouped table. 
Once this was done for both data frames, we then concatenated them into one larger combined data frame, which is critical for our initial and final model.

<iframe
  src="assets/combined_stats_sample.html"
  width="800"
  height="400"
  frameborder="0"
></iframe>

### Imputation
Because all of the team results had the relevant data that we needed for our analysis with no values missing, 
we found it unnecessary to impute any missing values. 
This was verified when making the scatter plots and models as if there were any missing values, 
there would have been glaring issues in the compilation step. 

## Framing a Prediction Problem
We are predicting whether or not a team wins a game based on the statistics we perceive to have the highest impact on winning: team kills per minute, the first to capture the mid-tower, 
the total gold of a team during the game, and the number of deaths the team accrued. 
This is a binary classification problem where the response variable is the outcome of the game, 
a win or a loss. We chose this problem because it represents a game result that can be influenced by both early game objectives, 
in this case, the first mid-tower as well as the cumulative performance of a team throughout the match. 
We will evaluate the model using the accuracy metric to help us determine the impact of these cumulative statistics on the prediction of a team’s match result. 
Accuracy would make more sense than an F-1 score because there is no real class imbalance (the number of wins and losses are exactly half) or a serious impact of Type I error or Type II error (it is only a video game, after all) within our predictions.

## Baseline Model
Our baseline model uses a logistical regression model that takes in the team kpm column, total gold, and first mid-tower columns, 
standardizes them by using the StandardScaler feature transform, 
and then fits the pipeline to the training data to train the model. 
The team kpm and total gold features are quantitative while the first midtower feature is ordinal (0 means the team did not get it, 
while 1 means the team did get the first midtower). The accuracy of this model is 86.1% for the large data frame, while it achieved 100% accuracy for the aggregated data frame. 
While we believe this model is already pretty solid, we do believe that there is room for improvement as we did not include the number of deaths within our pipeline, 
which we believe can improve the accuracy and serve as a crucial determining factor of how well a team defends itself against its enemies.

## Final Model
For our final model, we decided to add 'deaths' as a critical feature because we realized deaths not only give the enemy team gold 
but also temporary numerical advantages, making it a strong predictor of game results. We kept all the other features from our baseline model, and applied QuantileTransformer for 'team kpm', 'totalgold', and 'deaths' to handle their nonnormal distributions and potential outliers.
We chose a Decision Tree Classifier over the baseline logistic regression bcause we realized it can better capture non-linear relationships between the features we've chosen.
We used GridSearchCV with 5-fold cross-validation to tune our max-depth hyperparameter, testing values from 1 to 10, 
and finding the optimal max_depth was 6.
The final model achieved notiably better accuracy of 94.33% compared to the baseline model's 86.1%. 
The confusion matrix shows strong performance in indentifying wins and losses.
Our tree structure reveals that deaths <= 0.418 and total gold are crucial decision points in determining whether or not a team will win the match.
Our improvement in both accuracy and interpretability suggests that our model selection choice and feature engineering successfully captured the complex relationships between first midtower, total gold, deaths, and kpm in League of Legends game outcomes.
