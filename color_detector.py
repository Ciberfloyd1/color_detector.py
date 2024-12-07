import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Definir colores y sus rangos en HSV
COLOR_RANGES = {
    'red': ([0, 100, 100], [10, 255, 255]),
    'green': ([35, 100, 100], [85, 255, 255]),
    'blue': ([100, 100, 100], [130, 255, 255]),
    'yellow': ([20, 100, 100], [35, 255, 255]),
    'purple': ([130, 100, 100], [160, 255, 255]),
    'orange': ([10, 100, 100], [20, 255, 255]),
    'white': ([0, 0, 200], [180, 30, 255]),
    'black': ([0, 0, 0], [180, 255, 30]),
    'gray': ([0, 0, 30], [180, 30, 200])
}

def generate_color_data(num_samples=5000, image_size=64):
    X = []
    y = []
    
    for _ in range(num_samples):
        color_name = np.random.choice(list(COLOR_RANGES.keys()))
        label = list(COLOR_RANGES.keys()).index(color_name)
        
        lower, upper = COLOR_RANGES[color_name]
        hue = np.random.randint(lower[0], upper[0]+1)
        sat = np.random.randint(lower[1], upper[1]+1)
        val = np.random.randint(lower[2], upper[2]+1)
        
        image = np.full((image_size, image_size, 3), [hue, sat, val], dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        
        # Añadir ruido y variación
        noise = np.random.normal(0, 15, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        X.append(image)
        y.append(label)
    
    return np.array(X), np.array(y)

def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model():
    X, y = generate_color_data()
    X = X.astype('float32') / 255.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_model((64, 64, 3), len(COLOR_RANGES))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=15, validation_split=0.2, batch_size=32)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')
    
    model.save('color_model.h5')
    return history

def predict_color(image_path):
    if not os.path.exists('color_model.h5'):
        print("Model not found. Training a new model...")
        train_model()
    
    model = models.load_model('color_model.h5')
    colors = list(COLOR_RANGES.keys())
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    color_index = np.argmax(prediction)
    color = colors[color_index]
    confidence = prediction[0][color_index] * 100
    
    return color, confidence

def main():
    while True:
        image_path = input("Enter the path to your image (or 'q' to quit): ")
        if image_path.lower() == 'q':
            break
        
        if not os.path.exists(image_path):
            print("Image not found. Please enter a valid path.")
            continue
        
        color, confidence = predict_color(image_path)
        print(f"The predicted color is {color} with {confidence:.2f}% confidence.")

if __name__ == "__main__":
    main()
