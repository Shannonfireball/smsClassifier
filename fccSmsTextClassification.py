{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8RZOuS9LWQvv",
        "collapsed": true
      },
      "outputs": [],
      "source": [
        "# import libraries\n",
        "# 1. Uninstall the broken versions\n",
        "!pip uninstall -y tensorflow tf-nightly\n",
        "\n",
        "# 2. Install the specific stable version required by Colab's other libraries\n",
        "!pip install tensorflow==2.19.0\n",
        "\n",
        "# 3. Import and check\n",
        "import tensorflow as tf\n",
        "import pandas as pd\n",
        "from tensorflow import keras\n",
        "import tensorflow_datasets as tfds\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lMHwYXHXCar3",
        "collapsed": true
      },
      "outputs": [],
      "source": [
        "# get data files\n",
        "!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv\n",
        "!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv\n",
        "\n",
        "train_file_path = \"train-data.tsv\"\n",
        "test_file_path = \"valid-data.tsv\""
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "g_h508FEClxO"
      },
      "outputs": [],
      "source": [
        "train_df = pd.read_csv(train_file_path, sep='\\t', header=None, names=['label','message'])\n",
        "test_df = pd.read_csv(test_file_path, sep='\\t', header=None, names=['label','message'])\n",
        "print(test_df.head())\n",
        "print(train_df.head())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zOMKywn4zReN",
        "collapsed": true
      },
      "outputs": [],
      "source": [
        "train_df['label'] = train_df['label'].map({ 'ham': 0, \"spam\": 1,  })\n",
        "test_df['label'] = test_df['label'].map({ 'ham': 0, \"spam\": 1,  })\n",
        "print('values',train_df['label'].unique())"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.replace('', np.nan, inplace=True)\n",
        "test_df.replace('', np.nan, inplace=True)\n",
        "train_df.dropna(subset=['label', 'message'], inplace=True)\n",
        "test_df.dropna(subset=['label', 'message'], inplace=True)\n",
        "print(train_df.isnull().sum())\n",
        "print(test_df.isnull().sum())"
      ],
      "metadata": {
        "id": "IC9TeEoZ3x4Z"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "trainDataset = tf.data.Dataset.from_tensor_slices((train_df['message'].values, train_df['label'].values))\n",
        "testDataset = tf.data.Dataset.from_tensor_slices((test_df['message'].values, test_df['label'].values))\n",
        "\n",
        "trainDataset = trainDataset.batch(40).prefetch(tf.data.AUTOTUNE)\n",
        "testDataset = testDataset.batch(40).prefetch(tf.data.AUTOTUNE)"
      ],
      "metadata": {
        "id": "KMAJTfS2ygGm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "maxWords = 1000\n",
        "maxLength = 100\n",
        "encodeTestLayer = keras.layers.TextVectorization(max_tokens=maxWords, output_mode=\"int\", output_sequence_length=maxLength)\n",
        "encodeTestLayer.adapt(train_df['message'].values)"
      ],
      "metadata": {
        "id": "CMD3Icziz5qt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(len(train_df))\n",
        "print(len(test_df))\n"
      ],
      "metadata": {
        "id": "MUMnMFLQ80pA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "for text, label in trainDataset.unbatch().take(30):\n",
        "  print(f'label: {label.numpy()}, text: {text.numpy().decode('utf-8')[:50]}......')"
      ],
      "metadata": {
        "collapsed": true,
        "id": "s6UWKPJs-jHJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "tf.keras.preprocessing.sequence.pad_sequences"
      ],
      "metadata": {
        "id": "oqIurUfHCWK2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = keras.Sequential([\n",
        "    encodeTestLayer,\n",
        "    keras.layers.Embedding(maxWords, 64, mask_zero=True),\n",
        "    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, recurrent_dropout=0.01)),\n",
        "    keras.layers.Bidirectional(keras.layers.LSTM(64, recurrent_dropout=0.01)),\n",
        "    keras.layers.Dense(64, activation='relu'),\n",
        "    keras.layers.Dropout(0.2,),\n",
        "    keras.layers.Dense(1, activation='sigmoid'),\n",
        "])\n",
        "\n",
        "model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'],)\n",
        "model.fit(\n",
        "    trainDataset,\n",
        "    epochs=25,\n",
        "    verbose=1,\n",
        "    validation_data=testDataset\n",
        ")"
      ],
      "metadata": {
        "collapsed": true,
        "id": "CtKV7AXC65TG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "prediction = model.predict(tf.constant(['how are you doing today'], dtype=tf.string))\n",
        "predictedMessageType = 'spam' if prediction[0][0] >= 0.5 else 'ham'\n",
        "print(predictedMessageType)\n",
        "print(prediction[0][0])\n"
      ],
      "metadata": {
        "id": "C1N1FAYE-CEW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "J9tD9yACG6M9"
      },
      "outputs": [],
      "source": [
        "# function to predict messages based on model\n",
        "# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])\n",
        "def predict_message(pred_text):\n",
        "  prediction = model.predict(tf.constant([pred_text], dtype=tf.string))\n",
        "  predictedMessageType = 'spam' if prediction[0][0] >= 0.5 else 'ham'\n",
        "\n",
        "  return [ prediction[0][0].item(), predictedMessageType ]\n",
        "\n",
        "pred_text = \"how are you doing today?\"\n",
        "\n",
        "prediction = predict_message(pred_text)\n",
        "print(prediction)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Dxotov85SjsC"
      },
      "outputs": [],
      "source": [
        "# Run this cell to test your function and model. Do not modify contents.\n",
        "def test_predictions():\n",
        "  test_messages = [\"how are you doing today\",\n",
        "                   \"sale today! to stop texts call 98912460324\",\n",
        "                   \"i dont want to go. can we try it a different day? available sat\",\n",
        "                   \"our new mobile video service is live. just install on your phone to start watching.\",\n",
        "                   \"you have won £1000 cash! call to claim your prize.\",\n",
        "                   \"i'll bring it tomorrow. don't forget the milk.\",\n",
        "                   \"wow, is your arm alright. that happened to me one time too\"\n",
        "                  ]\n",
        "\n",
        "  test_answers = [\"ham\", \"spam\", \"ham\", \"spam\", \"spam\", \"ham\", \"ham\"]\n",
        "  passed = True\n",
        "\n",
        "  for msg, ans in zip(test_messages, test_answers):\n",
        "    prediction = predict_message(msg)\n",
        "    if prediction[1] != ans:\n",
        "      passed = False\n",
        "\n",
        "  if passed:\n",
        "    print(\"You passed the challenge. Great job!\")\n",
        "  else:\n",
        "    print(\"You haven't passed yet. Keep trying.\")\n",
        "\n",
        "test_predictions()\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "name": "fcc_sms_text_classification.ipynb",
      "private_outputs": true,
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {},
    "accelerator": "GPU"
  },
  "nbformat": 4,
  "nbformat_minor": 0
}