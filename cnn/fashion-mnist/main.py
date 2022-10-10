# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


if __name__ == "__main__":
    print(tf.__version__)
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(train_images.shape) # (60000, 28, 28)
    print(test_images.shape) # (10000, 28, 28)
    # plt.figure()
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()
    
    #将这些值缩小至 0 到 1 之间，然后将其馈送到神经网络模型。为此，请将这些值除以 255。
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1,28,28,1)

    print(train_images.shape)
    print("观察几个image")
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28, 28)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(10)
    # ])

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=2, padding="same"),
        tf.keras.layers.Conv2D(64, 3),
        tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=2, padding="same"),
        # tf.keras.layers.Conv2D(128, 3),
        # tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    print("进行训练...")
    print(model.summary())
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)
    print("评估准确率...")
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)
    print("进行预测...")
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print(predictions[0])
    print("分类:", class_names[np.argmax(predictions[0])])

    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()



