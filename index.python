# ==========================================
# 1. IMPORTAR BIBLIOTECAS
# ==========================================

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 2. CARREGAR DADOS - MNIST
# ==========================================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Dados carregados!")
print("Treino:", x_train.shape)
print("Teste:", x_test.shape)


# ==========================================
# 3. PRÉ-PROCESSAR OS DADOS
# ==========================================

# Normalizar os pixels para valores entre 0 e 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Adicionar o canal de cor
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# ==========================================
# 4. CONSTRUIR A REDE NEURAL CNN
# ==========================================

modelo = models.Sequential([
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(28, 28, 1)
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(
        128,
        activation='relu'
    ),

    layers.Dense(
        10,
        activation='softmax'
    )
])


# ==========================================
# 5. COMPILAR O MODELO
# ==========================================

modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ==========================================
# 6. TREINAR O MODELO
# ==========================================

historico = modelo.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test)
)


# ==========================================
# 7. AVALIAR O DESEMPENHO
# ==========================================

test_loss, test_acc = modelo.evaluate(
    x_test,
    y_test,
    verbose=2
)

print(f"\nPrecisão da IA: {test_acc * 100:.2f}%")


# ==========================================
# 8. SALVAR O MODELO
# ==========================================

modelo.save("modelo_mnist_completo.h5")

print("\nModelo salvo com sucesso!")


# ==========================================
# 9. CARREGAR O MODELO
# ==========================================

modelo_carregado = tf.keras.models.load_model(
    "modelo_mnist_completo.h5"
)


# ==========================================
# 10. FAZER UMA PREVISÃO
# ==========================================

amostra = x_test[0].reshape(1, 28, 28, 1)

predicao = modelo_carregado.predict(amostra)

numero_previsto = np.argmax(predicao)

print("Número previsto:", numero_previsto)
print("Número correto:", y_test[0])


# ==========================================
# 11. VISUALIZAR A AMOSTRA
# ==========================================

plt.imshow(
    x_test[0].reshape(28, 28),
    cmap="gray"
)

plt.title(f"Previsão: {numero_previsto}")

plt.axis("off")
plt.show()

