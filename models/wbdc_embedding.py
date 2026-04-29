# Import libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import pandas as pd

# Membuat model ANN untuk embedding fitur menggunakan Functional API
def create_model(X_train):
    input_dim = X_train.shape[1]  # Jumlah fitur

    # Input layer
    inputs = Input(shape=(input_dim,))

    # Beberapa hidden layers
    x = Dense(128, activation='relu')(inputs)  # Lapisan pertama
    x = Dense(64, activation='relu')(x)       # Lapisan kedua
    x = Dense(32, activation='relu')(x)       # Lapisan ketiga
    x = Dense(16, activation='relu')(x)       # Lapisan terakhir

    # Output layer (untuk klasifikasi biner: kanker ganas vs jinak)
    outputs = Dense(1, activation='sigmoid')(x)  # Untuk prediksi biner (0 = benign, 1 = malignant)

    # Buat model menggunakan Functional API
    model = Model(inputs, outputs)

    # Kompilasi model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Melatih model
def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Ekstraksi fitur (embedding) setelah pelatihan
def get_embeddings(model, X_train, X_test):
    # Ekstrak embedding dari lapisan ke-4 (setelah 3 hidden layers)
    embedding_model = Model(inputs=model.input, outputs=model.layers[4].output)  # Lapisan ke-4 (embedding)
    
    # Mendapatkan embedding untuk data latih dan uji
    train_embeddings = embedding_model.predict(X_train)
    test_embeddings = embedding_model.predict(X_test)

    print(f"\nShape of train_embeddings: {train_embeddings.shape}")
    print(f"Shape of test_embeddings: {test_embeddings.shape}")
    print(f"\nFitur Embedding untuk X_train (5 baris pertama):\n{train_embeddings[:5]}")

    return train_embeddings, test_embeddings

# Fungsi utama untuk menjalankan proses
def main():
    # Membaca file CSV dan mengonversinya ke numpy array
    X_train = pd.read_csv('data/preprocessed_wdbc/X_train.csv').values
    X_test = pd.read_csv('data/preprocessed_wdbc/X_test.csv').values
    y_train = pd.read_csv('data/preprocessed_wdbc/y_train.csv').values
    y_test = pd.read_csv('data/preprocessed_wdbc/y_test.csv').values

    # Membuat dan melatih model
    model = create_model(X_train)
    train_model(model, X_train, y_train, X_test, y_test)

    # Mendapatkan embedding
    train_embeddings, test_embeddings = get_embeddings(model, X_train, X_test)

    # Menyimpan embedding ke dalam file Numpy (.npy)
    np.save('data/embeddings/wbdc_train_embeddings.npy', train_embeddings)
    np.save('data/embeddings/wbdc_test_embeddings.npy', test_embeddings)

    print("Embeddings berhasil disimpan ke dalam format .npy.")

if __name__ == "__main__":
    main()