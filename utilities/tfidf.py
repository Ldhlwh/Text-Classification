from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ['我 来 自 上海 交通 大学',
	'我 来 自 北京',
	'我 从 北京 来 到 上海',
	'我 从 上海 回 到 北京',
	'交通 便利'
	]
vectorizer = TfidfVectorizer(min_df = 1)
vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
arr = vectorizer.fit_transform(corpus).toarray()

