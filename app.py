import nltk 
import re 
import streamlit as st 
import pickle


nltk.download('punkit')
nltk.download('stopwords')

#loaing clf and tfidf model 

clf = pickle.load(open('clf_xgboost.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(resume_text):
    cleantxt = re.sub('http\S+\s', ' ',resume_text)
    cleantxt = re.sub('@\S+', ' ',cleantxt)
    cleantxt = re.sub('#\S+\s', ' ',cleantxt)
    cleantxt = re.sub('RT|CC', ' ',cleantxt)
    cleantxt = re.sub('[%s]'% re.escape("""!"#$%&()*+<_>?:;<=>?@[\]^_'{|}~"""), ' ',cleantxt)
    cleantxt = re.sub(r'[^\x00-\x7f]', ' ',cleantxt)
    cleantxt = re.sub('\s+', ' ',cleantxt)
    return cleantxt

#webapp
def main():
        st.title('Resume Screening  Classify App')
        upload_file = st.file_uploader('Upload Your Resume', type=['txt','pdf'])

         
        
        if upload_file is not None:
            try:
                   resume_bytes = upload_file.read()
                   resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                   #if Unicode is fail try different code 
                   resume_text = resume_bytes.decode('latin-1')
            
            

            #########################for uploaded file ###########################################
            cleaned_resume = cleanResume(resume_text)
            cleaned_resume = tfidf.transform([cleaned_resume])
            prediction_id = clf.predict(cleaned_resume)[0]


            #map Id to category name 
            category_name = { 21:"Data Science", 
                            31: "HR",  
                            7: " Advocate'",  
                            8: "Arts", 
                            47: "Web Designing", 
                            36: "Mechanical Engineer", 
                            44: "Sales", 
                            33: "Health and fitness", 
                            18: "Civil Engineer", 
                            35: "Java Developer", 
                            14: "Business Analyst", 
                            43: "SAP Developer",  
                            9: "Automation Testing", 
                            27: "Electrical Engineering", 
                            38: "Operations Manager", 
                            41: "Python Developer", 
                            23: "DevOps Engineer",
                            37: "Network Security Engineer", 
                            39: "PMO", 
                            22: "Database", 
                            32: "Hadoop", 
                            26: "ETL Developer", 
                            24: "DotNet Developer", 
                            13: "Blockchain", 
                            46: "Testing", 
                            19: "DESIGNER", 
                            34: "INFORMATION-TECHNOLOGY", 
                            45: "TEACHER",  
                            1: "ADVOCATE", 
                            12: "BUSINESS-DEVELOPMENT", 
                            30: "HEALTHCARE", 
                            29: "FITNESS",  
                            2: "AGRICULTURE", 
                            11: "BPO",
                            42: "SALES", 
                            17: "CONSULTANT", 
                            20: "DIGITAL-MEDIA",  
                            5: "AUTOMOBILE", 
                            15: "CHEF", 
                            28: "FINANCE",  
                            3: "APPAREL", 
                            25: "ENGINEERING",  
                            0: "ACCOUNTANT", 
                            16: "CONSTRUCTION", 
                            40: "PUBLIC-RELATIONS", 
                            10: "BANKING", 
                            4: "ARTS",  
                            6: "AVIATION"
                
            }

            st.write('Prediction Category is :-',category_name.get(prediction_id))
        
if __name__=="__main__":
        main()
