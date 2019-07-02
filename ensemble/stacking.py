class Stacking:

	def __init__(self,models):
		self.reset_model(models)
		self.next_layer_train_table = None
		self.next_layer_test_table = None

	def reset_model(self,models):
		if not isinstance(models,dict):
			raise TypeError("models should be dict")
		self.models = []
		self.model_names = []
		for name, model in models.items():
			self.models.append(model)
			self.model_names.append(name)


	def fit(self,train_X,train_Y,test_X,*,num_folds=3,fold_sample_shuffle=False,is_classifier=True):
		# preprocess
		train_X = train_X.reset_index().drop("index",axis=1)
		train_Y = train_Y.reset_index().drop("index",axis=1)
		# make next layer train data
		if self.next_layer_train_table is None:
			next_layer_train_table = pd.DataFrame(index=train_X.index,columns=[])
			next_layer_test_table = pd.DataFrame(index=test_X.index,columns=[])
		else:
			next_layer_train_table = self.next_layer_train_table
			next_layer_test_table = self.next_layer_test_table
		# tmp variable for calculating start_row and end_row
		numrow = len(train_X)
		par_numrow = numrow // num_folds
		contain_extra_numrow = numrow % num_folds
		# KFold object
		fold = KFold(n_splits=num_folds,shuffle=fold_sample_shuffle)
		# in each model, fit the train data
		for model_id,model in enumerate(self.models):
			# initialize output data used in the next layer
			next_layer_train_data = np.zeros(numrow)
			next_layer_test_data = np.zeros((num_folds,len(test_X)))
			#
			start_row = 0
			# recode score
			score = 0.0
			#folding
			for fold_id in range(num_folds):
				#
				end_row = start_row + par_numrow
				end_row += 1 if fold_id < contain_extra_numrow else 0
				#make folded train table
				train_fold_X = train_X.loc[0:start_row]
				train_fold_X = train_fold_X.append(train_X.loc[end_row:])
				train_fold_Y = train_Y.loc[0:start_row]
				train_fold_Y = train_fold_Y.append(train_Y.loc[end_row:])
				#make folded test table
				test_fold_X = train_X.loc[start_row:end_row-1]
				test_fold_Y = train_Y.loc[start_row:end_row-1]

				# model fitting
				model.fit(train_fold_X,train_fold_Y)

				#we get train data used in next layer
				next_layer_train_data[start_row:end_row] = model.predict(test_fold_X)

				#calculate score
				score += model.score(test_fold_X, test_fold_Y)

				#make test data
				next_layer_test_data[fold_id] = model.predict(test_X)

				#update start_row
				start_row = end_row + 1

			# make next_layer_train_table
			if is_classifier:
				next_layer_train_data = next_layer_train_data.astype(int)
			next_layer_train_table[self.model_names[model_id]] = next_layer_train_data
			next_layer_train_table["TARGET"] = train_Y

			# make next_layer_test_table
			if is_classifier:
				next_layer_test_table[self.model_names[model_id]] = st.mode(next_layer_test_data,axis=0)[0][0].astype(int)
			else:
				next_layer_test_table[self.model_names[model_id]] = np.mean(next_layer_test_data,axis=0)

			# print score for this model
			print("### {} ### Score:{}".format(self.model_names[model_id],score / num_folds))

		self.next_layer_train_table = next_layer_train_table
		self.next_layer_test_table = next_layer_test_table

	def get_next_layer_table(self):
		return self.next_layer_train_table,self.next_layer_test_table




