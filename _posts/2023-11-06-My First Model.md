---
title: Launch & Learn - My First Model
date: 2023-11-06
categories: [Machine Learning, Python]
tags: [random-forest, kaggle]     # TAG names should always be lowercase
---

# Launch & Learn: My First Model
![Courtesy of Bing AI](/assets/my first model.jpg)
## Launch
After months of poking around Kaggle and looking into various machine learning models, or getting lost in documentation, I finally put together my first model and submission. What changed was simple: I realized worrying too much about perfection meant I never made any progress.

If you haven't heard of it, there's a fantastic course on machine learning I've been working through, called [fastai](https://course.fast.ai). An emphasis of the lessons there is a straight-forward suggestion to get something up and running so you can iterate and learn. After a few iterations I should see which levers are the most sensitive to improving (or degrading) the accuracy of my predictions.

This first model ranked 3328 out of 15382 submissions, somewhere in the ballpark of the 20th percentile. Not too shabby, but I'm sure accuracy can be increased since I went for expediency rather than accuracy when prepping the training data. More on that below.

Feel free to copy paste any of the below code to help yourself get started, or you can [check out this repo on my GitHub](https://github.com/ossiesf/titanic/blob/main/titanic-v1.ipynb).

## Learn
 We of course start with the basic provided first cell.
```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = '/kaggle/input/titanic'
train_data = pd.read_csv(path + '/train.csv')
train_data.head()
```

Now, to get started with preparing our data we can begin feature selection. Analyzing names seems like it could potentially impact accuracy negatively, as I don't think this data can give us a meaningful correlation (at least on this data set). The expedient thing to do was to simply drop that column. 
```python
# Create feature set X, along with our test data
X = train_data
test_data = pd.read_csv(path + '/test.csv')

# For the names, just dropping them for now
X = X.drop('Name', axis=1)
test_data = test_data.drop('Name', axis=1)
```

Here's where one interesting learning experience started: RandomForestRegressor cannot handle text input - it expects only numerical input. Since the 'Sex' feature should have a significant impact on model accuracy, it was important to include this feature in the training.

There were no numerical features to extract for this feature, however since there were only two options (gender inclusivity not being a thing in 1912) we can easily encode the data. By swapping the category with a numerical value, we can then proceed with training our model.

Fortunately, this can be easily done with pandas ['get_dummies()'](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) function. We can actually encode multiple features with one function call, so I included the 'Embarked' feature which has three potential text values of Q, C or S.

```python
# RandomForestRegressor expects numerical data, and cannot handle non-numeric categories
# For sex, embarked categories, we can encode them using pandas 'get_dummies'
# Which will create additional columns
X = pd.get_dummies(X, columns=['Sex', 'Embarked'])
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'])
```
The result of 'get_dummies()' above results in the addition of 5 columns in place of the original two: 2 for 'Sex', three for 'Embarked'. This allows the column to specify a true/false (or 1/0) value for each category. This is a technique known as One-Hot Encoding. See Chart 1 below.

```python
# Ticket & Cabin columns contain values that are difficult to encode
# Dropping those columns for now, but may revisit later
X = X.drop(['Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)

# Lastly, some rows are missing the age and fare fields, dropping those columns for now
X = X.drop(['Age', 'Fare'], axis=1)
test_data = test_data.drop(['Age', 'Fare'], axis=1)
# test_data = test_data.dropna(subset=['Age', 'Fare'])
X.head()
```

For the features 'Ticket' and 'Cabin', we have a lot of missing values as well as data that is not suitable for one-hot encoding. While these columns are valuable as they are related to the class of the passenger, there's a column indicating that information directly. While I imagine this results in some loss of accuracy, to get launched and iterate later, we will drop these columns.

| PassengerId | Pclass | SibSp | Parch | Sex_female | Sex_male | Embarked_C | Embarked_Q | Embarked_S |
|-------------|--------|-------|-------|------------|----------|------------|------------|------------|
| 0           | 1      | 3     | 1     | False      | True     | False      | False      | True       |
| 1           | 2      | 1     | 1     | True       | False    | True       | False      | False      |
| 2           | 3      | 3     | 0     | True       | False    | False      | False      | True       |
| 3           | 4      | 1     | 1     | True       | False    | False      | False      | True       |
| 4           | 5      | 3     | 0     | False      | True     | False      | False      | True       |

*Chart 1: Results of One-Hot Encoding*

Now we can create our targets, y, and drop that column from the training set. I gave this its own cell in Kaggle to make it easier to view the head of X or y.

```python
# Create y targets from the modified X dataframe
y = X['Survived']

# Drop the target column from X, which is already removed from test_data
X = X.drop('Survived', axis=1)
X.head()
```

 Now, we can use a RandomForestClassifier and fit it to our data. Note that there is both RandomForestRegressor as well as the classifier - since we have a binary choice of survived or perished, we need to classify the passengers into a category. There are no gradients of how 'survived' the passenger was.
```python
# Create random forest regressor
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 50, max_features = 'sqrt', max_depth = 5, random_state = 42).fit(X, y)
```

 Finally, we can leverage our model against `test_data` to get our predictions.
```python
# Get predictions
predictions = rf.predict(test_data)
predictions
```

 We will need to create a DataFrame object to hold our data. For our submission, we are required to submit only the PassengerId and Survived features.
```python
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
```

 This last line will spit out a csv file we can then submit as our predictions for a ranking.
```python
submission.to_csv('submission.csv', index=False)
```

I hope my learning process is helpful for anyone looking to get started with a basic machine learning model.

Happy Hacking!