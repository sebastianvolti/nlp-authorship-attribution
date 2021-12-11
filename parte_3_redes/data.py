import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from constants import TOTAL_FILES, TOTAL_TRAIN_FILES

def toLowerCase(text):
  return text.lower()

def removePunctuation(text):
  return re.sub('[\W_]+', ' ',text)

def removeRepetitions(text):
  def repl(matchobj):
    c=matchobj.group(0)
    return c[0]
  return re.sub(r'(\w)\1{2,}',repl ,text)         

def removen(text):
  return re.sub(r'\\n', "",text)

def RemoveStopWords(dataset):
  stop = stopwords.words('spanish')
  important_words = dataset['Paragraph'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
  return important_words


class Data():
  def __init__(self):
    buffer = ""
    buffer_train_list = []
    buffer_eval_list = []
    df_train_list = []
    df_eval_list = []
    train = pd.DataFrame(columns= ['Paragraph'])
    evaluation = pd.DataFrame(columns= ['Paragraph'])
    authors = ['Conrad', 'Zola', 'Proust', 'Austen', 'Flaubert']

    for author_id, author in enumerate(authors):
      train_df = pd.DataFrame(columns= ['Paragraph'])
      eval_df = pd.DataFrame(columns= ['Paragraph'])
      total = TOTAL_FILES[author]
      total_train = TOTAL_TRAIN_FILES[author]
      for elem in range(total):
        file_name = author + str(elem + 1) 
        df = pd.read_csv(f'parte_3_redes/resources/{file_name}utf8.txt', engine='python', names=["Paragraph"], sep="\t")
        for index, row in df.iterrows():
          buffer += row['Paragraph'] + "\n"
        if elem <= total_train:
          buffer_train_list.append(buffer)
        else:
          buffer_eval_list.append(buffer)
        buffer = ""

      train_df['Paragraph'] = buffer_train_list
      train_df['Author'] = author_id

      eval_df['Paragraph'] = buffer_eval_list
      eval_df['Author'] = author_id

      df_train_list.append(train_df)
      df_eval_list.append(eval_df)
      buffer_train_list.clear()
      buffer_eval_list.clear()

    train = pd.concat(df_train_list, ignore_index=True)
    evaluation = pd.concat(df_eval_list, ignore_index=True)
    print("train dataset\n")
    print(train)

    for i , row in train.iterrows():
      train.at[i,'Paragraph'] = self.preprocess(row["Paragraph"])
    
    for i , row in evaluation.iterrows():
      evaluation.at[i,'Paragraph'] = self.preprocess(row["Paragraph"])

    self.train = train
    self.val = evaluation


  def preprocess(self, text):
    text = removen(text)
    text = toLowerCase(text)
    return text

def main():
  pass

if __name__ == "__main__":
  main()