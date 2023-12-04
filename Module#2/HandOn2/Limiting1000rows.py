from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import PorterStemmer
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
def get_and_clean_data():
    data = pd.read_csv('../Resource/software_developer_united_states_1971_20191023_1.csv')
    description = data['job_description']
    cleaned_description = description.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    cleaned_description = cleaned_description.apply(lambda s: s.lower())
    cleaned_description = cleaned_description.apply(lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), '')))
    cleaned_description = cleaned_description.drop_duplicates()
    return cleaned_description



# limit to just 1000 rows
cleaned_description = get_and_clean_data()[:1000]

# replace non alphabeths with spaces, and collapse spaces
cleaned_description = cleaned_description.apply(lambda s: re.sub(r'[^A-Za-z]', ' ', s))
cleaned_description = cleaned_description.apply(lambda s: re.sub(r'\s+', ' ', s))

# tokenize
tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))

# remove stop words
stop_dict = set(stopwords.words())
sw_removed_description = tokenized_description.apply(lambda s: set(s) - stop_dict)
sw_removed_description = sw_removed_description.apply(lambda s: [word for word in s if len(word) > 2])


#create stem caches
concated = np.unique(np.concatenate([s for s in tokenized_description.values]))
stem_cache = {}
ps = PorterStemmer()
for s in concated:
    stem_cache[s] = ps.stem(s)

#stem
stemmed_description = sw_removed_description.apply(lambda s: [stem_cache[w] for w in s])

# print(tokenized_description)

cv = CountVectorizer(analyzer=lambda x: x)
# vectoriser = cv.fit
X = cv.fit_transform(stemmed_description)
print(pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out()))
