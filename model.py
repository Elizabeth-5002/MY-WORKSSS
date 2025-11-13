import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Emotion labels (7 classes)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to load and preprocess image for prediction
def preprocess_image(image_path, target_size=(224, 224)):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Function to predict emotion
def predict_emotion(model, image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions)
    confidence = float(predictions[0][emotion_index])
    return EMOTIONS[emotion_index], confidence

# Build/Load the pre-trained model
def create_or_load_model(model_path='model.h5'):
    if os.path.exists(model_path):
        # Load existing model
        model = tf.keras.models.load_model(model_path)
        print("Loaded pre-trained model.")
        return model
    
    # Else, create a simple transfer learning model (minimal "training" step)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base layers
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(len(EMOTIONS), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Optional: Train on a small sample (uncomment if you download FER2013)
    # from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # # Assume data_dir = 'path/to/FER2013/train' with subfolders for emotions
    # datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    # train_gen = datagen.flow_from_directory('path/to/FER2013/train', target_size=(224, 224), batch_size=32, subset='training')
    # val_gen = datagen.flow_from_directory('path/to/FER2013/train', target_size=(224, 224), batch_size=32, subset='validation')
    # model.fit(train_gen, epochs=5, validation_data=val_gen)
    
    # Save the model
    model.save(model_path)
    print("Model created and saved.")
    return model

# For pre-trained: We'll download a community fine-tuned model (run this once)
# Go to https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4
# But for simplicity, the above creates a base one. For better accuracy, replace with:
# hub_url = "https://tfhub.dev/tensorflow/tf2-preview/mobilenet_v2/feature_vector/4"
# But we'll stick to pure Keras for ease.

if __name__ == "__main__":
    model = create_or_load_model()
    # Test with a sample image (provide your own path)
    # emotion, conf = predict_emotion(model, 'path/to/test_image.jpg')
    # print(f"Predicted: {emotion} ({conf:.2f})")