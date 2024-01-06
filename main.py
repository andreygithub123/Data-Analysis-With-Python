import numpy as np
import pandas as pd
import utils as utl
import classes.class_PCA as pca
import graphics as g

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

# start the PCA procedure
obs=data.index.values
vars = data.columns.values[1:]
matrix_values=data_2[vars].values
#print(matrix_values,type(matrix_values))

#standardize matrix values then turn it into pandas.Dataframe
matrixstd=utl.standardize(matrix_values)
matrixstd_df=pd.DataFrame(data=matrixstd,index=obs,columns=vars)
#matrixstd_df.to_csv('./dataOUT/matrixstd.csv')
#matrixstd_df.to_excel('./dataOUT/matrixstd.xlsx')

#variance-covariance matrix => check the PCA CLASS
#create a PCA object
modelPCA = pca.PCA(matrixstd)

#eigen values and eigen vectors on the variance-covariance matrix ( in PCA class )
alpha=modelPCA.getEigenValues()
g.principalComponents(eigenvalues=alpha)
g.show() # right now the graphic is incorect bc our eigenvalues are not descending order

#fetch the principal components
C = modelPCA.getPrinComp();
components = ['C'+str(j+1) for j in range(C.shape[1])]
C_df=pd.DataFrame(data=C,index=obs,columns=components)
#save the principal components ins s CSV file
C_df.to_csv('./dataOUT/PrinComp.csv')

# fetch the correlation factors ( factor loadings )
Rxc = modelPCA.getFactorLoadings()
# save the factor loadings into a CSV file
Rxc_df = pd.DataFrame(data=Rxc,index=vars,columns=components)
g.correlogram(matrix=Rxc_df,title='Correlogram of factor loadings')
g.show()


#fetch the scores ( standardized principal components )
scores =  modelPCA.getScores()
# save the score into a CSV file
scores_df = pd.DataFrame(data=scores, index = obs, columns=components)
g.correlogram(matrix=scores_df,title="Correlogram of scores")
g.show()


#fetch the quality of observation on the reprezentation
qualityObs = modelPCA.getQualitObs()
# save the qualiity of observation into a CSV file
qualityObs_df = pd.DataFrame(data=qualityObs, index=obs,columns=components)
g.correlogram(matrix=qualityObs_df,title='Correlogram of quality of observations')

# fetch hte observation contributions to the axes variance
contribObs = modelPCA.getContribObs()
# save the obs
contribObs_df = pd.DataFrame(data = contribObs,index = obs,columns=components)
g.correlogram(matrix=contribObs_df, title='Observation contributions to the axes variance')
g.show()

##comonalities how many PCA the observations have in common
#matrix with elements partial sum : n1=n1 n2 = n1+n2 ; nm =n1+n2+...+nm
# fetch the commonalities
common = modelPCA.getCommonalities()
#save the commonalities into a CSV file
common_df=pd.DataFrame(data=common,index=vars, columns=components)