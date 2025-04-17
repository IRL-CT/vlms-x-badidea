#imports
import numpy as np
import pandas as pd
import os

#load data
data = pd.read_csv('../data/all_video_descriptions.csv')

#for each row, in column 'description', remove all descriptions matching the following terms:
remove_list = ["In the video, ",
]
#a lot more modifications here

#clean data
for ind, row in data.iterrows():
    description = row['DESCRIPTION']
    for term in remove_list:
        description = description.replace(term, "")
    
    #remove quotation marks
    description = description.replace('"', "")
    #print(description)
    #save all as a string
    data.loc[ind, 'DESCRIPTION'] = str(description)


#save cleaned data
data.to_csv('../data/all_video_descriptions_cleaned.csv', index=False)

print('Data cleaned and saved to data/all_video_descriptions_cleaned.csv')


