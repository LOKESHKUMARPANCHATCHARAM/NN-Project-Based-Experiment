#Project Based Experiments
## Objective :
 Build a Multilayer Perceptron (MLP) to classify handwritten digits in python
## Steps to follow:
## Dataset Acquisition:
Download the MNIST dataset. You can use libraries like TensorFlow or PyTorch to easily access the dataset.
## Data Preprocessing:
Normalize pixel values to the range [0, 1].
Flatten the 28x28 images into 1D arrays (784 elements).
## Data Splitting:

Split the dataset into training, validation, and test sets.
Model Architecture:
## Design an MLP architecture. 
You can start with a simple architecture with one input layer, one or more hidden layers, and an output layer.
Experiment with different activation functions, such as ReLU for hidden layers and softmax for the output layer.
## Compile the Model:
Choose an appropriate loss function (e.g., categorical crossentropy for multiclass classification).Select an optimizer (e.g., Adam).
Choose evaluation metrics (e.g., accuracy).
## Training:
Train the MLP using the training set.Use the validation set to monitor the model's performance and prevent overfitting.Experiment with different hyperparameters, such as the number of hidden layers, the number of neurons in each layer, learning rate, and batch size.
## Evaluation:

Evaluate the model on the test set to get a final measure of its performance.Analyze metrics like accuracy, precision, recall, and confusion matrix.
## Fine-tuning:
If the model is not performing well, experiment with different architectures, regularization techniques, or optimization algorithms to improve performance.
## Visualization:
Visualize the training/validation loss and accuracy over epochs to understand the training process. Visualize some misclassified examples to gain insights into potential improvements.

# Program:
```
pip install tensorflow matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

tf.random.set_seed(42)
np.random.seed(42)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train_flat = x_train.reshape((-1, 28*28))
x_test_flat = x_test.reshape((-1, 28*28))
num_classes = 10
y_train_cat = utils.to_categorical(y_train, num_classes)
y_test_cat = utils.to_categorical(y_test, num_classes)

x_train_flat, x_val_flat, y_train_cat, y_val_cat, y_train, y_val = train_test_split(
    x_train_flat, y_train_cat, y_train, test_size=0.12, random_state=42, stratify=y_train
)

def build_mlp(input_dim=784, hidden_layers=[512, 256], dropout_rate=0.3, l2_reg=1e-4):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

es = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
chkpt_path = "mlp_mnist_best.h5"
mc = callbacks.ModelCheckpoint(chkpt_path, monitor='val_loss', save_best_only=True)

model = build_mlp(hidden_layers=[512, 256], dropout_rate=0.3, l2_reg=1e-4)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(x_train_flat, y_train_cat,
                    validation_data=(x_val_flat, y_val_cat),
                    epochs=20,
                    batch_size=128,
                    callbacks=[es, mc],
                    verbose=2)

model.save("mlp_mnist_model.h5")

test_loss, test_acc = model.evaluate(x_test_flat, y_test_cat, verbose=0)
print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")

y_pred_probs = model.predict(x_test_flat)
y_pred = np.argmax(y_pred_probs, axis=1)


print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.colorbar()
plt.xticks(range(10))
plt.yticks(range(10))
plt.tight_layout()
plt.show()

mis_idx = np.where(y_pred != y_test)[0]
corr_idx = np.where(y_pred == y_test)[0]

def plot_examples(indices, title, n=9):
    plt.figure(figsize=(7,7))
    chosen = np.random.choice(indices, size=min(n, len(indices)), replace=False)
    for i, idx in enumerate(chosen):
        plt.subplot(3,3,i+1)
        plt.imshow(x_test[idx], cmap='gray')
        plt.axis('off')
        plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if len(mis_idx) > 0:
    plot_examples(mis_idx, "Misclassified examples (True: T, Predicted: P)", n=9)
plot_examples(corr_idx, "Correctly classified examples (True: T, Predicted: P)", n=9)

print("Saved: mlp_mnist_model.h5 and mlp_mnist_best.h5")
```

## Output:



