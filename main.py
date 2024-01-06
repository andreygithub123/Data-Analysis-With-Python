import numpy as np
import pandas as pd
import utils as utl

# dataframe of our data
data=pd.read_excel('./dataIN/aaDate.xlsx',index_col=0)
#print(data,type(data))

# in our excel Highway km column is not completed and in the dataframe it apears at NaN
label=['Highway km']
#print(data[label])

# now we need NaN verification and replacement


# we are trying to do the replaceNaN only on 1 column to see if it works
NaNColumn=data[label] # - Highway km
#print(NaNColumn,type(NaNColumn))

# converting data.Series into a nparray
numpycolumn=NaNColumn.values
#print(numpycolumn,type(numpycolumn))

# calling the function => we see that it works properly : the results is still a numpy array
utl.replaceNAN(numpycolumn)
#print(numpycolumn,type(numpycolumn))

#now we have to refactor it back into a pandas.Dataframe
#data.columns[7]  -- asociated with highway km
result_df_column=pd.DataFrame(numpycolumn,columns=[data.columns[7]])
print(result_df_column,type(result_df_column))


#now we re doing it on the whole dataframe
data_copy=data.copy()
indexes=data_copy.index
#print(data_copy,type(data_copy))
data_values=data_copy.values
utl.replaceNAN(data_values)
result_df=pd.DataFrame(data_values,columns=data_copy.columns)
result_df.set_index(indexes,inplace=True)
print(result_df,type(result_df))
#result_df.to_excel('./dataOUT/aaDateRefactored.xlsx')


#this is directly for dataframes
data_2=data.copy()
utl.replaceNAN(data_2)
#print(data_2,type(data_2))
#data_2.to_excel('./dataOUT/aaDateRefactoredEasierMethod.xlsx')