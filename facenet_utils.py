from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np

KNN_CLASSIFIER_DICT = {"trained": False, "classifier": None}

def load_dataset(dataset_path):
	dataset_embeddings = np.load(dataset_path)
	faces_embeddings, labels = dataset_embeddings['arr_0'], dataset_embeddings['arr_1']
	faces_embeddings = faces_embeddings.reshape(-1,128)
	faces_embeddings = normalize_vectors(faces_embeddings)
	return faces_embeddings, labels

def normalize_vectors(vectors):
	# normalize input vectors
	normalizer = Normalizer(norm='l2')
	vectors = normalizer.transform(vectors)

	return vectors

def labels_encoder(labels):
	# label encode targets: one-hot encoding
	# this is needed by machine learning classifiers
	out_encoder = LabelEncoder()
	out_encoder.fit(labels)
	labels = out_encoder.transform(labels)
	return out_encoder, labels
	
	
def predict_using_classifier(faces_embeddings, labels, face_to_predict_embedding, threshold=45):

	classifier = get_classifier(faces_embeddings, labels, "knn")

	out_encoder, labels = labels_encoder(labels)
	# print(labels)
	face_to_predict_embedding = normalize_vectors(face_to_predict_embedding)
	# faces_embedding = normalize_vectors(faces_embeddings)

	face_to_predict_embedding = face_to_predict_embedding[0]

	# prediction for the face
	samples = np.expand_dims(face_to_predict_embedding, axis=0)
	# print(samples.shape)
	# If error raised check using dataset embeddings and classifier data are the same, remove classifier files
	yhat_class = classifier.predict(samples)
	# print(yhat_class)
	yhat_prob = classifier.predict_proba(samples)
	class_index = yhat_class[0]
	class_probability = (yhat_prob[0, class_index] * 100)

	#print('class_index: ',class_index)
	# print('class_probability',class_probability)
	
	if class_probability >= threshold:
		predicted_name = out_encoder.inverse_transform(yhat_class)[0]
	else:
		predicted_name = 'Unknown'

	
	# predicted_name += f' {class_probability}'

	# print("predicted_name ", predicted_name, " class_probability ", class_probability)
	return predicted_name, class_probability

def get_classifier(faces_embeddings=None, labels=None, classifier_name='knn'):
	'''
			The method will load the classifier if it exist in the specified path
			otherwise, it will train the classifier using faces_embeddings and labels
	'''

	if (KNN_CLASSIFIER_DICT["trained"] == False):
		#print('[INFO] Training Classifier...')
		classifier = train_classifier(faces_embeddings, labels, classifier_name=classifier_name)
		KNN_CLASSIFIER_DICT["classifier"] = classifier
		KNN_CLASSIFIER_DICT["trained"] = True
		# print(CLASSIFIER_DICT[classifier_name]["trained"])
	else:
		#print('[INFO] Trained classifier assigned...')
		classifier = KNN_CLASSIFIER_DICT["classifier"]

	#print('[INFO] Classifier Training Done...')

	return classifier

def train_classifier(faces_embeddings, labels, classifier_name='knn'):
	'''
			The method will train the classifier. There are three classifiers:
			- KNN
			- SVM
			- Neural Network
	'''
	# print('---------------',faces_embeddings.shape)
	faces_embeddings = normalize_vectors(faces_embeddings)
	out_encoder, labels = labels_encoder(labels)

	if classifier_name == 'knn':
		classifier = KNeighborsClassifier(n_neighbors=100, p=1, weights="distance", metric="euclidean")
	else:
		raise ValueError('Classifier name not found, classifier should be: knn, svm, neural_network')

	# fit model
	classifier.fit(faces_embeddings, labels)

	# save the classifier to file
	save_path = os.path.join(os.getcwd(), 'classifiers',
							 '{}.sav'.format(classifier_name))
	joblib.dump(classifier, save_path)
	print('Classifier Saved to [%s]...' % save_path)
	return classifier

def extract_face_from_image(input_file_path, detector):
	# load image from file
	image = Image.open(input_file_path)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = np.asarray(image)
	# detect faces in the image
	results = detector.detect_faces(pixels)

	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize((160, 160))
	face_array = np.asarray(image)
	
	return face_array
