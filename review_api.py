# load the model
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

model=joblib.load('Review_model.mod')
data=pd.read_csv('Review_test_data.csv')
#test_data=pd.DataFrame()
#ref=pd.DataFrame()
# creating object for PorterStemmer
ps=PorterStemmer()

def Review_classification(test):
	#test_data=pd.read_csv(filename)
	#ref=pd.read_csv(filename)
	df=data_clean_step1(test)
	X=df['Text'].values
	y_pred=model.predict(X)
	return y_pred
    
# Writing method to clean data
def data_clean_step1(data_set):
	corpus=[]
	for i in range(0,len(data_set)):
		# Removing all the words other than alphabet
		review=re.sub("[^a-zA-Z]"," ",str(data_set['Text'][i]))

		# Converting into lowercase
		review=review.lower()

		#Splitting review as words
		review=review.split()

		# Stemming
		review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

		# Joining words (making sentences with words stem)
		review=' '.join(review)

		# Making list of reviews
		corpus.append(review)
      
	# replacing processed reviews 
	for i in range(len(corpus)):
		data_set['Text'][i]=corpus[i]

	return data_set
    
def main():
	st.title('Review classifier using Natural Language Processing')
	st.write('This app is to identify the reviews where the semantics of review text does not match rating.')
	st.write('We need to upload a "csv" file of following "format" to use the app and click on "Classify" button.')
	st.write(data.head(4))
	
	html_temp="""
    <div style="background-color:tomato;padding:10px;">
    <h2 style="color:white;text-align:center;">Review classifier using Natural Language Processing</h2>
    </div>
    """
	st.markdown(html_temp,unsafe_allow_html=True)
	
	st.subheader("Select CSV file to classify reviews")
	filename = st.file_uploader("Upload a file", type=("csv"))
	if filename is not None:
		try:
			if st.button('Classify'):
				test_data=pd.read_csv(filename)
				ref_data=test_data.copy(deep=True)
				y_pred=Review_classification(test_data)
				review_ID=[]
				for i in range(len(y_pred)):
					if ( (y_pred[i]==1)and (ref_data['Star'][i]<2)):
						review_ID.append(ref_data['ID'][i])
				result=ref_data[ref_data['ID'].isin(review_ID)]
				result.reset_index(inplace=True)
				result=result.iloc[:,1:]
				st.subheader('Classified Reviews')
				st.write('Reviews where the semantics of review text does not match rating.')				
				st.write(result)
		except:
			st.error('Please choose a file')
			
			
  
if __name__=='__main__':
	main()
    