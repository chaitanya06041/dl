OpenCV Face Rocognition(Using Image) (Assignment 2)
# Import necessary libraries
from cv2 import imread, CascadeClassifier, rectangle, waitKey, destroyAllWindows
from google.colab.patches import cv2_imshow
import cv2

# Load the image file
image_path = '/content/d66510ee-983b-11ef-a8b0-0242ac11000f-removebg.png'  # Replace with your image path
pixels = imread(image_path)

# Check if the image is loaded properly
if pixels is None:
    print("Image not found. Please check the path.")
else:
    # Load the pre-trained face detection model
    classifier = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Perform face detection
    bboxes = classifier.detectMultiScale(pixels, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Print and draw bounding box for each detected face
    for box in bboxes:
        # Extract coordinates
        x, y, width, height = box
        x2, y2 = x + width, y + height
        
        # Draw a rectangle around each detected face
        rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 2)

    # Display the image with detected faces in Colab
    cv2_imshow(pixels)

    # Release any resources
    destroyAllWindows()


________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________




OPENCV FACE RECOGNITION(USING LIVE CAMERA) (Assignment 2)




!pip install opencv-python tensorflow
import cv2
import base64
import numpy as np
from PIL import Image
import io
from IPython.display import display, Javascript
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow




# JavaScript code to capture a photo using the webcam
js_code = '''
function initCamera() {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.style.display = 'none';
        document.body.appendChild(video);
        const streamPromise = navigator.mediaDevices.getUserMedia({video: true});
        streamPromise.then((stream) => {
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                resolve(video);
            };
            video.play();
        }).catch((error) => {
            reject(error);
        });
    });
}

async function takePhoto() {
    const video = await initCamera();
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const img = canvas.toDataURL('image/jpeg');
    return img;
}
'''

display(Javascript(js_code))

# Function to convert JavaScript captured image to OpenCV format
def js_to_image(js_reply):
    image_bytes = base64.b64decode(js_reply.split(',')[1])
    image_PIL = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image_PIL)
    frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return frame

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture image using the webcam
    js_reply = eval_js('takePhoto()')
    frame = js_to_image(js_reply)

    if frame is None:
        continue

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Display the image with detected faces
    cv2_imshow(frame)

    # Break the loop if 'Esc' key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()








______________________________________________________________________________________________________________________________________________________________










IMAGE CLASSIFICATION USING CNN(Assignment 3)



import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Classes in CIFAR-10
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Visualize the first few images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i][0]])
plt.show()

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # Output layer for 10 classes
])



# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Train the model
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Plot training history
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()





from tensorflow.keras.preprocessing import image
import numpy as np


def predict_external_image(image_path):
    # Load the image
    img = image.load_img(image_path, target_size=(32, 32))  # Resize to model's input size
    plt.imshow(img)  # Display the image
    plt.axis('off')
    plt.show()
    
    # Convert image to array and preprocess
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    predictions = probability_model.predict(img_array)
    predicted_label = np.argmax(predictions[0])
    confidence = 100 * np.max(predictions[0])
    
    print(f"Predicted class: {class_names[predicted_label]}")
    print(f"Confidence: {confidence:.2f}%")
    return predicted_label

# Example: Predict an external image
image_path = "/content/download (1).jpg"  # Replace with your image path
predict_external_image(image_path)




_______________________________________________________________________________________________________________________________________________________________



YOLO OBJECT DETECTION (Assignment 7)




pip install torch torchvision opencv-python
!git clone https://github.com/ultralytics/yolov5
!cd yolov5 && pip install -r requirements.txt

import torch
import cv2
import matplotlib.pyplot as plt

# Load YOLOv5 model (pretrained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the image
img_path = '/content/dog.jpg'  # Replace with the path to your image
img = cv2.imread(img_path)

# Perform object detection
results = model(img)

# Display results
results.show()  # This will show the image with bounding boxes and labels.

# Optionally, save the result image
results.save('output/')  # Save the results with bounding boxes

# To get more detailed results, such as class labels, confidence scores, and bounding boxes:
df = results.pandas().xyxy[0]  # Results as pandas dataframe
print(df)








_______________________________________________________________________________________________________________________________________________________________



OCR TEXT EXTRACTION (Assignment 5)







import numpy as np
import matplotlib.pyplot as plt
# Install pytesseract and Tesseract-OCR
!pip install pytesseract
!apt-get update
!apt-get install -y tesseract-ocr
!pip install google-colab

# Import modules
import cv2
import numpy as np
import pytesseract
# from google.colab.patches import cv2_imshow
# from google.colab import files


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    denoised = cv2.fastNlMeansDenoising(gray)       # Remove noise
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Binary thresholding
    return thresh


    def detect_text_regions(image):
    # Detecting words
    boxes = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)  # Get text region data
    return boxes

    def draw_bounding_boxes(image, boxes):
    output = image.copy()  # Copy original image to draw boxes
    n_boxes = len(boxes['level'])  # Total number of detected text regions
    for i in range(n_boxes):
        if int(boxes['conf'][i]) > 60:  # Only consider boxes with confidence > 60%
            (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])  # Box coordinates
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle on the image
    return output


    def get_detected_text(boxes):
    detected_text = []
    n_boxes = len(boxes['level'])  # Total number of detected boxes
    for i in range(n_boxes):
        if int(boxes['conf'][i]) > 60:  # Only extract text with confidence > 60%
            detected_text.append(boxes['text'][i])  # Append detected text
    return ' '.join(detected_text)  # Combine text into a single string




from google.colab import files

# Upload the image
uploaded = files.upload()

image = cv2.imread("download.jpg")


# Check if image was successfully read
if image is None:
    print(f"Error: Unable to read the image file: {image_path}")
else:
    # Display original image
    print("\nOriginal Image:")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.show()

    # Preprocess the image and detect text regions
    preprocessed = preprocess_image(image)  # Preprocessing
    boxes = detect_text_regions(preprocessed)  # Detect text regions

    # Draw bounding boxes on original image
    image_with_boxes = draw_bounding_boxes(image, boxes)

    # Display the image with bounding boxes
    print("\nImage with Text Detection Regions:")
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.show()

    # Extract and print the detected text
    detected_text = get_detected_text(boxes)
    print("\nDetected Text:")
    print(detected_text)

________________________________________________________________________________________________________________________________________________________________







Sentiment Analysis (Assignment 6)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess the data
max_features = 10000  # Vocabulary size
max_len = 200         # Maximum sequence length

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Build the LSTM model
model = Sequential([
    Embedding(input_dim=max_features, output_dim=128, input_length=max_len),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')


def predict_sentiment(text):
    # Preprocess the input text
    word_index = imdb.get_word_index()
    words = text.lower().split()
    sequence = [word_index.get(word, 0) + 3 for word in words]  # +3 to adjust for special tokens in IMDb
    sequence = pad_sequences([sequence], maxlen=max_len)

    # Predict the sentiment
    prediction = model.predict(sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f"Predicted Sentiment: {sentiment} (Confidence: {prediction:.4f})")

# Example usage
predict_sentiment("This movie was fantastic and highly enjoyable!")
predict_sentiment("The plot was boring and the acting was terrible.")





________________________________________________________________________________________________________________________________________________________________




Feedforward Neural Network (Assignment 1)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

# Get data and split into train and test datasets
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Scale down values of pixels from 0-255 to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Visualizing data
print(train_images.shape)
print(test_images.shape)
print(train_labels)

# Display first image
plt.imshow(train_images[0], cmap='gray')
plt.show()

# Defining neural network model
my_model = tf.keras.models.Sequential()
my_model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
my_model.add(tf.keras.layers.Dense(128, activation='relu'))  # Rectified Linear Unit
my_model.add(tf.keras.layers.Dense(10, activation='softmax'))  # Softmax for multi-class classification

# Compiling the model
my_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model
my_model.fit(train_images, train_labels, epochs=3)

# Checking model accuracy on test data
val_loss, val_acc = my_model.evaluate(test_images, test_labels)
print('Test accuracy: ', val_acc)

# Predicting on a new image
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28), color_mode="L")  # Load the image in grayscale
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input
    img_array = img_array / 255.0  # Normalize the image

    # Get model prediction
    predictions = my_model.predict(img_array)
    predicted_label = np.argmax(predictions)  # Get the class with the highest probability

    return predicted_label, predictions

# Test the prediction function with a new image
image_path = 'path_to_image.png'  # Replace with the path to the image you want to predict
predicted_label, predictions = predict_image(image_path)

# Displaying the predicted result
print(f'Predicted Label: {predicted_label}')
print(f'Predictions: {predictions}')



________________________________________________________________________________________________________________________________________________________________


Stock Price Prediction using RNN (Assignment 4)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv('/content/TSLA.csv')  # Ensure the dataset has 'Date' and 'Close' columns
dataframe['Date'] = pd.to_datetime(dataframe['Date'])
dataframe.set_index('Date', inplace=True)

closing_prices = dataframe[['Close']].values
scaler_obj = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler_obj.fit_transform(closing_prices)

def generate_sequences(data, seq_len):
    X_seq, y_seq = [], []
    for idx in range(len(data) - seq_len):
        X_seq.append(data[idx:idx + seq_len])
        y_seq.append(data[idx + seq_len])
    return np.array(X_seq), np.array(y_seq)

sequence_len = 60  # You can modify the sequence length as needed
X_features, y_labels = generate_sequences(normalized_data, sequence_len)

train_test_split_idx = int(len(X_features) * 0.8)
X_train_data, y_train_data = X_features[:train_test_split_idx], y_labels[:train_test_split_idx]
X_test_data, y_test_data = X_features[train_test_split_idx:], y_labels[train_test_split_idx:]

rnn_model = Sequential()
rnn_model.add(SimpleRNN(units=50, activation='relu', input_shape=(sequence_len, 1)))
rnn_model.add(Dense(1))  # Output layer for predicting the stock price

rnn_model.compile(optimizer='adam', loss='mean_squared_error')

training_history = rnn_model.fit(X_train_data, y_train_data, epochs=10, batch_size=32, validation_data=(X_test_data, y_test_data))
     
future_predictions = []
recent_sequence = normalized_data[train_test_split_idx - sequence_len:train_test_split_idx]

for _ in range(20):  # Predict for the next 20 days
    recent_sequence = np.reshape(recent_sequence, (1, sequence_len, 1))  # Ensure correct shape for RNN
    predicted_price = rnn_model.predict(recent_sequence)

    future_predictions.append(predicted_price[0][0])  # Store predicted price

    # Reshape predicted price and append it to the sequence
    predicted_price = np.reshape(predicted_price, (1, 1, 1))
    recent_sequence = np.append(recent_sequence[:, 1:, :], predicted_price, axis=1)
     
predicted_future_prices = scaler_obj.inverse_transform(np.array(future_predictions).reshape(-1, 1))
     
actual_future_prices = scaler_obj.inverse_transform(normalized_data[train_test_split_idx:train_test_split_idx+20])

plt.plot(dataframe.index[train_test_split_idx:train_test_split_idx+20], actual_future_prices, label='Actual Prices')
plt.plot(dataframe.index[train_test_split_idx:train_test_split_idx+20], predicted_future_prices, label='Predicted Prices')
plt.title('Actual vs Predicted Stock Prices for the Next 20 Days')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

for day in range(20):
    print(f"Day {day+1}: Actual: {actual_future_prices[day][0]}, Predicted: {predicted_future_prices[day][0]}")
     

________________________________________________++++++++++++________________________________++++++++++++_______________________++++++++++++++++++++___________________________________________________________________________________________________________________________________________________________________________




