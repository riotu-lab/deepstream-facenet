from sklearn.preprocessing import Normalizer
from PIL import Image
import numpy as np

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

	
def predict_using_classifier(faces_embedding, labels, face_to_predict_embedding):
    class_probability = None
    out_encoder, labels = labels_encoder(labels)
    # print(labels)
    face_to_predict_embedding = normalize_embeddings(
        face_to_predict_embedding, normalization_technique='l2')
    faces_embedding = normalize_embeddings(
        faces_embedding, normalization_technique='l2')

    face_to_predict_embedding = face_to_predict_embedding[0]

    # prediction for the face
    samples = np.expand_dims(face_to_predict_embedding, axis=0)
    # print(samples.shape)
    # If error raised check using dataset embeddings and classifier data are the same, remove classifier files
    yhat_class = classifier.predict(samples)
    # print(yhat_class)
    yhat_prob = classifier.predict_proba(samples)
    # print(yhat_prob)
    class_index = yhat_class[0]
    #print('class_index: ',class_index)
    class_probability = (yhat_prob[0, class_index] * 100)
    # print('class_probability',class_probability)
    predicted_name = out_encoder.inverse_transform(yhat_class)[0]

    # print("predicted_name ", predicted_name, " class_probability ", class_probability)
    return predicted_name

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
