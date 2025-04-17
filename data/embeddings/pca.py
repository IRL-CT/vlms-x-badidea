#SCRIPT to do PCA on text embeddings csv 
#import
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA



def main():
        
    #load the csv file
    df = pd.read_csv('./clip_text_embeddings.csv')
    print(df.head())

    #select only columsn 2:end
    
    df_cols_first = df.iloc[:,0:2]
    df = df.iloc[:,2:]
    print('DF COLS FIRST', df_cols_first.head())

    #do PCA to keep 95% of the variance
    pca = PCA(n_components=0.95)
    pca.fit(df)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())
    #see how many components we have
    print(pca.n_components_)

    #transform the data
    df_pca = pca.transform(df)

    #save the pca data as csv
    full_df = pd.concat([df_cols_first, pd.DataFrame(df_pca)], axis=1, ignore_index=True)
    #rename the columns
    new_cols = ['VIDEO', 'TIME_START'] + [f'PC{i}' for i in range(1, df_pca.shape[1] + 1)]
    full_df.columns = new_cols
    #check for nan
    print('checking for nan values')
    print(full_df.isnull().sum())
    print(full_df.shape)
    #remove rows with nan values
    #full_df = full_df.dropna()
    #new index
    full_df = full_df.reset_index(drop=True)
    full_df.to_csv('clip_text_embeddings_pca.csv', index=False)

    #save the pca model
    import joblib
    joblib.dump(pca, 'pca_model.pkl')


if __name__ == '__main__':
    main()
