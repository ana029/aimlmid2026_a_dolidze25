# aimlmid2026_a_dolidze25
Task1_1.Finding the correlation
Find the data at the following address "max.ge/aiml_midterm/51892_html". On the given online graph, the data is displayed
with blue dots. When hovering the mouse over the data, the coordinates of the data point are displayed on the screen.
Find Pearson's correlation coefficient and describe the process in your report. (5 points).
The report must also include a relevant graph for visualization. (5 points).

Given Data On the given online graph,
The dataset consists of 10 paired observations obtained from the online graph (blue dots):

x = [−9, −7, −5, −3, −1, 1, 3, 5, 7, 9]

y = [7.9, 6.7, 4.8, 5.1, 0.9, −0.5, −2.8, −4.7, −6.5, −8.3]

To measure the linear relationship between variables x and y, Pearson’s correlation coefficient (r) was calculated using python code.
The value r = −0.9914 indicates a very strong negative linear correlation between x and y.
This means that as x increases, y decreases almost linearly. The points closely follow a downward-sloping straight line, which confirms the strength of the relationship.
# Pearson Correlation Analysis

This is the correlation analysis of the dataset extracted from the online graph.

![Scatter Plot](correlation_plot.png)

2. Spam email detection
   Given the data file at the following address "max.ge/aiml_midterm/a_dolidze25_51892_csv" with email features and its classes (spam or legitimate). The main goal of this task is to develop one Python console application for email classification within spam and legitimate classes. Your program should do the actions described below. You should provide the corresponding data in the report as described below:
   1. Upload the provided data file to your repository and provide a link to the uploaded file in your report. (1 point).
      The dataset used in this project was provided by the course and uploaded to this repository.
**Dataset link:**  
[data/a_dolidze25_51892.csv](a_dolidze25_51892.csv)

The dataset contains numerical features extracted from emails and a binary class label indicating whether the email is spam or legitimate.

     2.  Your application should create and train a logistic regression model on 70% of this data (2 points). Provide a link to the appropriate source code(s) in the report (1 point). In the report, provide and describe the data loading and processing code (2 points), the model used (with the code) for logistic regression (1 point), and provide the coefficients found by your model (1 point).
Data Loading and Processing

The dataset is loaded using the pandas library.  
The target variable is `is_spam`, where:
- 1 indicates spam email
- 0 indicates legitimate email

All other columns are used as input features.

```python
data = pd.read_csv("a_dolidze25_51892.csv")
data.head()
from sklearn.model_selection import train_test_split
X = data[['words', 'links', 'capital_words', 'spam_word_count']]
y = data['is_spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
The dataset was split into training and testing subsets using a 70/30 ratio.

A Logistic Regression classifier from scikit-learn was used to solve the binary classification problem.

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

Coefficients 

```python
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})
coefficients
```
   3. Within your application validate (find the Confusion Matrix and Accuracy) your model on the data that you have not used for training (1 point). Present the Confusion Matrix and Accuracy in the report and describe the code for finding them. (2 points).
The trained model was evaluated on the test dataset using Accuracy and Confusion Matrix.

```python
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
```

   
   4. Your application should have ability to check email text i.e. parse it, extract the same features that you have in provided data file and evaluate it for spam using your model. (3 points).
