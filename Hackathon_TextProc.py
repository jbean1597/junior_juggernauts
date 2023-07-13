#%%
import pandas as pd
import nltk 
import re

data = pd.read_json("arxiv_data_add.json")
# %%
data.info()
# %%
data['text'] = data['text'].astype('string')
data['text'] = data['text'].str.replace('\n', ' ').str.replace('\r', '').str.replace('- ', '').str.replace('-', ' ')
# %%
df_copy = data.copy()
df_copy.info()
#%%
df_copy['text'][5]
#%%
# Cleaning
pattern1 = r'References |R eferences |Acknowledgements |A cknowledgements '
pattern2 = r'Abstract |ABSTRACT |A bstract |A BSTRACT |1 Abstract |1 ABSTRACT |1 A bstract |1 A BSTRACT |Introduction |I ntroduction |INTRODUCTION |I NTRODUCTION |1 Introduction |1 INTRODUCTION |1 I ntroduction |1 I NTRODUCTION'

for i, text in enumerate(df_copy['text']):
    text = text[72:] # Remove arxiv tag
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    ref_text = re.split(pattern1, text) # Remove references
    intro_text = re.split(pattern2, ref_text[0], maxsplit=1)
    if len(intro_text) > 1:
        df_copy['text'][i] = intro_text[1]
    else:
        print(i)


# %%
df_copy['text'][5]
# %%
df_copy.head(6)
# %%
df_copy.to_json('small_arxiv_cleaned.json')
# %%
