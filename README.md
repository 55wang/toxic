Open Project: A Social Media Application
Deadline: 16 Apr 2018(Mon, 1800 Hrs)

### Environment Setting
1. Python 2.7
2. anaconda 5.1
3. sqlite local database

### Installation
pip install -r requirements.txt

### Usage
1. Run 'data_preprocess_v2.4.py' to prepocess tweet content, including data cleaning (e.g., remove url, punctuations, time), word tokenize, stemming, remove low-frequency words and stopping words.

2. Run 'data_model V2.2.py', which adopts a few classifiers and ensemble model. You are supposed to see performances (classification score, average percision, average recall) printing into the screen.

3. Run the main webapp 'microblog.py'

4. visit http://127.0.0.1:5000 to view the web app

### Folder Structure
code folder contains all the python code
data folder contains all the input and output
app folder contains all the web MVC components