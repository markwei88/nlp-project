import re
import string
import nltk
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 

class Preprocess:
    def __init__(self, text):
        self.text = text

    def deal_with_others(self):
        self.text = [re.sub(r'\\u[\S]+','',sent) for sent in self.text]
        self.text = [re.sub(r'http://[\S]+','',sent) for sent in self.text]
        return self.text

    def remove_punctuation(self):
        punctuation = string.punctuation + '+——！，。？、~@#￥%……&*（）【】'
        self.text = [sent.translate(str.maketrans(punctuation, ' '*len(punctuation))) for sent in self.text]
        return self.text

    def stemming(self):
        st=PorterStemmer()
        steem = [[st.stem(word) for word in sent.split()] for sent in self.text]
        X_stem = []
        for sent in steem:
            context = ''
            for word in sent :
                context = context + ' ' + word
            X_stem.append(context)
        self.text = X_stem
        return self.text

    def lemmatizer(self):
        lemmatizer = WordNetLemmatizer()  
        lemmatizer.lemmatize('leaves') 
        lemma = [[lemmatizer.lemmatize(word) for word in sent.split()] for sent in self.text]
        X_lemma = []
        for sent in lemma:
            context = ''
            for word in sent :
                context = context + ' ' + word
            X_lemma.append(context)
        self.text = X_lemma
        return self.text

    def segmentation(self):
        from ekphrasis.classes.segmenter import Segmenter
        seg_eg = Segmenter(corpus = "english")
        seg_tw = Segmenter(corpus="twitter")
        self.text = [seg_tw.segment(sent) for sent in self.text]
        return self.text

def datastories_processor(x):
    from ekphrasis.dicts.emoticons import emoticons
    from ekphrasis.classes.tokenizer import SocialTokenizer
    from ekphrasis.classes.preprocessor import TextPreProcessor

    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
                'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )

    x = [text_processor.pre_process_doc(sent) for sent in x]
    temp = []
    for sent in x:
        context = ''
        for word in sent :
            context = context + ' ' + word
        temp.append(context)

    return temp

