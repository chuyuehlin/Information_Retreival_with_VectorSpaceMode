from pprint import pprint
from Parser import Parser,Parser_ch
import util
import os
import glob
import math
import nltk
import jieba

class VectorSpace:
	""" A algebraic model for representing text documents as vectors of identifiers. 
	A document is represented as a vector. Each dimension of the vector corresponds to a 
	separate term. If a term occurs in the document, then the value in the vector is non-zero.
	"""

	#Collection of document term vectors
	documentVectors = []
	documentVectors_tf_idf = []

	#Mapping of vector index to keyword
	vectorKeywordIndex=[]

	#Tidies terms
	parser=None
	
	IDFVector = []

	ischinese=False

	def __init__(self, documents=[], ischinese=False):
		self.documentVectors=[]
		self.ischinese=ischinese

		if ischinese == False :
			self.parser = Parser()
		elif ischinese == True :
			self.parser = Parser_ch()
		
		if(len(documents)>0):
			self.build(documents)


	def build(self,documents):
		""" Create the vector space for the passed document strings """
		self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
		self.IDFVector = self.getIDFVector(documents)
		self.documentVectors = [self.makeVector(document) for document in documents]
		self.documentVectors_tf_idf = [[a*b for a,b in zip(self.IDFVector,documentVector)] for documentVector in self.documentVectors]
	
	def getVectorKeywordIndex(self, documentList):
		""" create the keyword associated to the position of the elements within the document vectors """
		#Mapped documents into a single word string vocabularyString = " ".join(documentList)
		vocabularyString = " ".join(documentList)
		
		#(English mode)in tokenise function, vocabularyString will be removed punctuations, stemmed, splited to a list of words.
		#(Chinese mode)in tokenise function, vocabularyString will be removed punctuations, segmented to a list of words by jieba. 
		vocabularyList = self.parser.tokenise(vocabularyString)
			
		#Remove common words which have no search value		 
		vocabularyList = self.parser.removeStopWords(vocabularyList)
		uniqueVocabularyList = util.removeDuplicates(vocabularyList)

		vectorIndex={}
		offset=0
		#Associate a position with the keywords which maps to the dimension on the vector used to represent this word		 
		for word in uniqueVocabularyList:
			vectorIndex[word]=offset
			offset+=1
		
		return vectorIndex	#(keyword:position)

	def getIDFVector(self, documentList):

		count = [0] * len(self.vectorKeywordIndex)

		for doc in documentList:
			docstring = self.parser.tokenise(doc)
			uniquedocstring = util.removeDuplicates(docstring)		
			for word in uniquedocstring:
				if self.vectorKeywordIndex.get(word)!= None:
					count[self.vectorKeywordIndex[word]] += 1
		IDF = [math.log(len(documentList)/word) for word in count]

		return IDF


	def makeVector(self, wordString,no_tokenise=False):
		""" @pre: unique(vectorIndex) """

		#Initialise vector with 0's
		vector = [0] * len(self.vectorKeywordIndex)
		if no_tokenise == False:
			# if wordString is a document or an English input query, it has to be tokenize
			wordList = self.parser.tokenise(wordString)
		elif no_tokenise == True:
			# if wordString is a chinese input query, we don't have to tokenize again couse it has been split after input
			wordList = wordString
		for word in wordList:
			#some of words are stop words, so they may not exist in vectorKeywordIndex
			if self.vectorKeywordIndex.get(word)!= None:
				vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
		

		return vector


	def buildQueryVector(self, termList,only_NV=False):
		""" convert query string into a term vector """
		# boolean value only_NV is used to determine whether the function should ignore non-nounce and non-verb words.
		if only_NV == True:
			pos_tags = nltk.pos_tag(termList)
			tags = set(['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ'])
			termList = [word for word,pos in pos_tags if pos in tags]
			query = self.makeVector(termList, no_tokenise=True)

		if only_NV == False:
			query = self.makeVector(termList,self.ischinese)

		return query


	def related(self,documentId):
		""" find documents that are related to the document indexed by passed Id within the document Vectors"""
		ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
		return ratings


	def search(self,searchList,sim_func="cosine",TF_IDF=False):
		""" search for documents that match based on a list of terms """
		# set sim_func "cosine" to calculate Cosine similarity
		# set sim_func "euclidean" to calculate Euclidean Distance. The smaller the distance, the higher the similarity.
		# boolean value TF_IDF is used to determine whether take IDF into similarity calculation
		queryVector = self.buildQueryVector(searchList)
		
		if sim_func == "cosine" :
			if(TF_IDF==True):
				ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors_tf_idf]
			elif(TF_IDF==False):
				ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
		
		elif sim_func == "euclidean" :
			if(TF_IDF==True):
				ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors_tf_idf]
			elif(TF_IDF==False):
				ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]

		return ratings

	def search_relevance_feedback(self,original,feedback):
		""" search for documents that match based on a list of terms """
		originalVector = self.buildQueryVector(original)
		feedbackVector = self.buildQueryVector(feedback,only_NV=True)
		queryVector = [ o+f*0.5 for o,f in zip(originalVector,feedbackVector)]
		ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors_tf_idf]
		
		return ratings

def read_files(path):
	documents = []
	doc_name = []
	for filename in glob.glob(os.path.join(path, '*.txt')):
		documents.append(open(filename, 'r').read())
		doc_name.append(filename[14:24])
	return documents, doc_name

if __name__ == '__main__':
	
	mode = int(input("Please input a number from 1 to 3, input others to exit the programe:  "))
	if mode == 1 or mode == 2:
		#read txt files
		
		documents, doc_name = read_files("./EnglishNews")
		
		vectorSpace = VectorSpace(documents)
		
		input_keywords = input('Please input a query in ENGLISH: ')
		
		#######################################################################################
		# (English documents and English input query)  					                      #
		# mode1 - list top10 documents relevant to input query in 4 ways: 					  #
		# 1. Term Frequency (TF) Weighting + Cosine Similarity            					  #
		# 2. Term Frequency (TF) Weighting + Euclidean Distance           					  #
		# 3. TF-IDF Weighting + Cosine Similarity                         					  #
		# 4. TF-IDF Weighting + Euclidean Distance		                  					  #
		#######################################################################################

		if mode == 1:
			
			print("\n--------------------")
			print("\nTerm Frequency (TF) Weighting + Cosine Similarity\n")
			score_vector = list(zip(doc_name, vectorSpace.search(input_keywords,"cosine")))
			sorted_by_second = sorted(score_vector, key=lambda tup: tup[1],reverse = True)
			print("File names  Scores")
			for top10 in sorted_by_second[:10]:
				print(top10[0]," %.5f" %top10[1] )	
			
			print("\n--------------------")
			print("\nTerm Frequency (TF) Weighting + Euclidean Distance\n")
			score_vector = list(zip(doc_name, vectorSpace.search(input_keywords,"euclidean")))
			sorted_by_second = sorted(score_vector, key=lambda tup: tup[1])
			print("File names  Scores")
			for top10 in sorted_by_second[:10]:
				print(top10[0]," %.5f" %top10[1] )	
			
			print("\n--------------------")
			print("\nTF-IDF Weighting + Cosine Similarity\n")
			score_vector = list(zip(doc_name, vectorSpace.search(input_keywords,"cosine",TF_IDF=True)))
			sorted_by_second = sorted(score_vector, key=lambda tup: tup[1],reverse = True)
			print("File names  Scores")
			for top10 in sorted_by_second[:10]:
				print(top10[0]," %.5f" %top10[1] )	
			
			print("\n--------------------")
			print("\nTF-IDF Weighting + Euclidean Distance\n")
			score_vector = list(zip(doc_name, vectorSpace.search(input_keywords,"euclidean",TF_IDF=True)))
			sorted_by_second = sorted(score_vector, key=lambda tup: tup[1])
			print("File names  Scores")
			for top10 in sorted_by_second[:10]:
				print(top10[0]," %.5f" %top10[1] )	
			print("\n--------------------")
		
		#######################################################################################
		# mode2 - 																			  #
		# Use Relevance Feedback to list top10 documents relevant to input query.             #
		# Document vector weighted by TF-IDF and use Cosine Similarity as similarity function #
		#######################################################################################
		
		if mode ==2:
			print("\n--------------------")
			print("\nTF-IDF Weighting + Cosine Similarity\n")
			score_vector = list(zip(doc_name, vectorSpace.search(input_keywords,"cosine",TF_IDF=True)))
			sorted_by_second = sorted(score_vector, key=lambda tup: tup[1],reverse = True)
			feedback_query = open("./EnglishNews/" + sorted_by_second[0][0] + ".txt", 'r').read()				
			score_vector = list(zip(doc_name, vectorSpace.search_relevance_feedback(input_keywords,feedback_query)))
			sorted_by_second = sorted(score_vector, key=lambda tup: tup[1],reverse = True)
			print("File names  Scores")
			for top10 in sorted_by_second[:10]:
				print(top10[0]," %.5f" %top10[1] )	
			print("\n--------------------")	
	
		#######################################################################################
		# (Chinese doucments and Chinese input query)										  #
		# mode3 - list top10 documents relevant to input query in 2 ways:					  #
		# 1. Term Frequency (TF) Weighting + Cosine Similarity								  #
		# 2. TF-IDF Weighting + Cosine Similarity											  #
		#######################################################################################
	
	if mode == 3:
		
		documents, doc_name = read_files("./ChineseNews")
		
		vectorSpace = VectorSpace(documents,ischinese=True)
	
		# in Chinese mode, we do not tokenize input query but split with " " after input
		input_keywords = input('Please input a query in CHINESE: ').split()
		
		print("\n--------------------")
		print("\nTerm Frequency (TF) Weighting + Cosine Similarity\n")
		score_vector = list(zip(doc_name, vectorSpace.search(input_keywords,"cosine",TF_IDF=False)))
		sorted_by_second = sorted(score_vector, key=lambda tup: tup[1],reverse = True)
		print("File names  Scores")
		for top10 in sorted_by_second[:10]:
			print(top10[0]," %.5f" %top10[1] )	

		print("\n--------------------")
		print("\nTF-IDF Weighting + Cosine Similarity\n")
		score_vector = list(zip(doc_name, vectorSpace.search(input_keywords,"cosine",TF_IDF=True)))
		sorted_by_second = sorted(score_vector, key=lambda tup: tup[1],reverse = True)
		print("File names  Scores")
		for top10 in sorted_by_second[:10]:
			print(top10[0]," %.5f" %top10[1] )	
		print("\n--------------------")
			

###################################################
