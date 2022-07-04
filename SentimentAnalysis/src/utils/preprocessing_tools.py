from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import TweetTokenizer

from emot.emo_unicode import EMOTICONS_EMO


def label_encoder(series, encoded):
  """ Helper function to convert numeric labels to encoded vectors """
  key= series['label'].iloc[0]
  series['lbl_1'] = encoded[key][0]
  series['lbl_2'] = encoded[key][1]
  series['lbl_3'] = encoded[key][2]
  return series


def custom_split(sentence, tk = TweetTokenizer()):
  # Use tokenize method to split sentences
  sentence = tk.tokenize(sentence)
  return sentence

# Function for converting emoticons into word
def convert_emoticons(text):
    for emot in EMOTICONS_EMO:
      text = text.replace(emot,EMOTICONS_EMO[emot])
    return text


def text_normalization(word_, 
                       stemmer = SnowballStemmer('english').stem, 
                       lemmatizer = WordNetLemmatizer().lemmatize):
  word_ = stemmer(word_)
  word_ = lemmatizer(word_)
  return word_