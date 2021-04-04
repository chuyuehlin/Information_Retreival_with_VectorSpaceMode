#http://tartarus.org/~martin/PorterStemmer/python.txt
from PorterStemmer import PorterStemmer
import jieba
#jieba.dt.cache_file = 'jieba.cache.new'
class Parser:

	#A processor for removing the commoner morphological and inflexional endings from words in English
	stemmer=None

	stopwords=[]

	def __init__(self):
		
		self.stemmer = PorterStemmer()
		#English stopwords from ftp://ftp.cs.cornell.edu/pub/smart/english.stop
		self.stopwords = open('stop_word_en.txt', 'r').read().split()


	def clean(self, string):
		""" remove any nasty grammar tokens from string """
		puncs = ["!",";",",",":",".","?"]
		for punc in puncs: 
			string = string.replace(punc," ") 
		
		puncs = ["(",")","[","]","{","}","'",'"',"<",">"]
		for punc in puncs: 
			string = string.replace(punc,"") 
		string = string.replace("\n"," ")
		string = string.replace("\s+"," ")
		string = string.lower()
		return string

	def removeStopWords(self,List):
		""" Remove common words which have no search value """
		return [word for word in List if word not in self.stopwords ]


	def tokenise(self, string):
		""" break string up into tokens and stem words """
		string = self.clean(string)
		words = string.split(" ")
		
		return [self.stemmer.stem(word,0,len(word)-1) for word in words]


class Parser_ch:

	#A processor for removing the commoner morphological and inflexional endings from words in English

	stopwords=[]

	def __init__(self):
		
		self.stopwords = open('stop_word_ch.txt', 'r').read().split()

	def clean(self, string_list):
		puncs = ["》","《"," ","，","。","：","；","！","？","「","」","『","』","、","（","）"]
		string_list = [word for word in string_list if word not in puncs]		
		return string_list
	

	def removeStopWords(self,list):
		""" Remove common words which have no search value """
		return [word for word in list if word not in self.stopwords ]


	def tokenise(self, string):
		""" break string up into tokens and stem words """
		string = string.replace("\n"," ")
		string_list = jieba.lcut(string)
		string_list = self.clean(string_list)
		return string_list



