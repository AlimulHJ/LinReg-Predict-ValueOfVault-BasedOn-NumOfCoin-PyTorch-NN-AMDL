### Regression - predict value of vaults based on the number of gold & silver coins in each vault.

# Library used:
# 1. From PyTorch(torch) <-- nn(Neural Network)
2. From pytest <-- For testing

# Ex-3: Regression in Harry Potter Universe
## Overview
In this exercise, we will use regression to predict the value of vaults in a hypothetical scenario based on the number of gold and silver coins in each vault.

## Objectives
1. Train a model using given data of vaults.
2. Use the trained model to predict the value of other vaults.

## The Assignment
We are in the Harry-Potter-verse. 
We have the data about the value of several vaults in GB Pounds.
We also know how many pieces of gold there are in each vault.
Here is a list.
you don't have to copy it, a function in the tests will generate it. 

| # Gold | # Silver | Value (GBP)  |
|-------|------|---------|
| 24.0, | 2.0, | 1422.40 |
| 24.0, | 4.0, | 1469.50 |
| 16.0, | 3.0, | 1012.70 |
| 25.0, | 6.0, | 1632.20 |
| 16.0, | 1.0, |  952.20 |
| 19.0, | 2.0, | 1117.70 |
| 14.0, | 3.0, |  906.20 |
| 22.0, | 2.0, | 1307.30 |
| 25.0, | 4.0, | 1552.80 |
| 12.0, | 1.0, |  686.70 |
| 24.0, | 7.0, | 1543.40 |
| 19.0, | 1.0, | 1086.50 |
| 23.0, | 7.0, | 1495.20 |
| 19.0, | 5.0, | 1260.70 |
| 21.0, | 3.0, | 1288.10 |
| 16.0, | 6.0, | 1111.50 |
| 24.0, | 5.0, | 1523.10 |
| 19.0, | 7.0, | 1297.40 |
| 14.0, | 4.0, | 946.40  |
| 20.0, | 3.0, | 1197.10 |

We would like to guess the value of a few other vaults, given the number of gold and silver coins known to be in them.

### Part 1. Train a model 
* Using 1d or 2d data, (i.e. using just the gold pieces counter or both coin cunter), create a model and train it. 
* The tests will pass when the loss is small enough.
* You need to change some of the code in the file `regression.py` - follow what we did in class.
* You can get a tuple with the shape of a tensor using `shape`

### Part 2. Use the model to predict some unknown values
* Once you've trained the model, these tests will predict some unseen vaults.
* Thought excercise - ungraded, how would you calculate the modeled value of a gold coin, given a trained model?

---

## Validating and Evaluating Your Results

### Online
1. After committing and pushing your code, check the mark on the top line (near the commit ID).
2. If some tests are failing, click on the ❌ to open up a popup, which will show details about the errors.
3. You can click the [Details]() link to see what went wrong. Pay special attention to lines with the words "Failed" or "error".

![screnshot](images/details_screenshot.png)

4. Near the bottom of the [Details]() page, you can see your score. Here are examples of 0/5 and 5/5:

![score](images/score.png) ![success](images/success.png)

5. When you achieve a perfect score, you will see a green checkmark near the commit ID.

![green](images/green.png)

### Locally
1. You can test your code locally by installing and running `pytest` (`pip install pytest` or `conda install pytest`).
2. Run the tests using the command `pytest` in your terminal. This will show the status of each test and any errors that occurred.

Good luck!