TEAM AJI INTERNATIONAL

We used a model combination technique using the following base models:

1. 5-gram interpolated Kneser-Ney Language Model, using the KenLM language model from cdec/MOSES, trained on a random 20% of the following:
	a. APNews corpus 
	b. NYT corpus
	c. AFPNews corpus
   
   We used the same 'unkification' scheme as the Berkeley parser, taking into account whether an unknown word ends in 'ly', 'ing', and check whether it contains digits, dashes, etc.After sampling a random 20% from each corpus, we concatenate the sampled corpora and train the aforementioned language model on that. It results in approximate 400-500 million tokens

2. Class-Factored LSTM language model, with dropout and decaying SGD learning rate. We used the Brown cluster to obtain the word cluster and the same 'unkification' scheme as the Ken-LM language model. The LSTM language model was trained on the following:

	a. A random 0.1% of the APNews corpus (~900,000 tokens)
	b. Another random 0.2% of the APNews corpus (~2 million tokens)
	c. All gold reference translations on the provided training and test data

The last training data (all gold reference translations) are used to make the model learn what a good translation in the domain should look like.

3. METEOR and BLEU, including all of these variants:

	a. Basic METEOR
	b. Basic METEOR run separately on the original, stemmed, lowercased, and tagged data
	c. The METEOR run with interpolated weights on the original, stemmed, lowercased, and tagged data (as a smoothing factor)
	d. Basic sentence level BLEU (didn't work well on its own)
	e. Separate BLEU for the original, stemmed, lowercased, and tagged data
	f. BLEU run with interpolated weights on the original, stemmed, lowercased, and tagged data (as a smoothing factor)

We used cross-validation and hyper-parameter search to get the best performance out of each individual model

4. We do model combination using a classifier that takes the prediction of each baseline model as features, using both linear multi-class SVM, RBF Kernel multi-class SVM, and random forest. SVM (whether linear or RBF, depending on the feature choice) work best. We use manual feature selection and SVM hyper-parameter search using a held-out dev set and also cross-validation.
