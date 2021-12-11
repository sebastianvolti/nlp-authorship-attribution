import os
import pandas as pd

authors = ['Conrad', 'Zola', 'Proust', 'Austen', 'Flaubert']

total_files = {
  "Austen": 130,
  "Conrad": 110,
  "Flaubert": 123,
  "Proust": 174,
  "Zola": 108,
}

for author_id, author in enumerate(authors):
    total = total_files[author]
    for elem in range(total):
        file_name = author + str(elem + 1) 
        os.system(f'iconv -f WINDOWS-1252 -t UTF-8 resources/{file_name}.txt > {file_name}utf8.txt'
)
