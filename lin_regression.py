import numpy as np
import pandas as pd
import statsmodels.api as sm
# import matplotlib.pyplot as plt 



loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

# Clean Interest Rate: 
cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
loansData['Interest.Rate'] = cleanInterestRate

# Clean Loan Length:
cleanLoanLength = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))
loansData['Loan.Length'] = cleanLoanLength

# Clean FICO Range
cleanFicoRange = loansData['FICO.Range'].map(lambda x: x.split('-'))
cleanFicoRange = cleanFicoRange.map(lambda x: [int(num) for num in x])
loansData['FICO.Range'] = cleanFicoRange

# print loansData['FICO.Range'].head()
# 81174    [735, 739]
# 99592    [715, 719]
# 80059    [690, 694]
# 15825    [695, 699]
# 33182    [695, 699]

# Create a New Column called FICO.Score: 
new_column = loansData['FICO.Range'].map(lambda x: x.pop(0)) #How calc midpoint? 
loansData['FICO.Score'] = new_column


# Set 3 Cleaned DataFrame Columns to variables: 
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# NUMPY
# The dependent variable
y = np.matrix(intrate).transpose()
# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

# COLUMN_STACK METHOD
x = np.column_stack([x1,x2])

# print x (LOWER CASE x)
# [[  735 20000]
#  [  715 19200]
#  [  690 35000]
#  ..., 
#  [  680 10000]
#  [  675  6000]
#  [  670  9000]]

# CREATE LINEAR MODEL: 
X = sm.add_constant(x)

# print X (UPPER CASE X)
# [[  1.00000000e+00   7.35000000e+02   2.00000000e+04]
#  [  1.00000000e+00   7.15000000e+02   1.92000000e+04]
#  [  1.00000000e+00   6.90000000e+02   3.50000000e+04]
#  ..., 
#  [  1.00000000e+00   6.80000000e+02   1.00000000e+04]
#  [  1.00000000e+00   6.75000000e+02   6.00000000e+03]
#  [  1.00000000e+00   6.70000000e+02   9.00000000e+03]]

# STATS MODEL: 
model = sm.OLS(y,X)

f = model.fit()

# I can't get this to print the summary:
f.summary()











