import tensorflow as tf

import iris_data


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
# (train_x, train_y), (test_x, test_y) = iris_data.load_data()
train, test = iris_data.load_data()
features_train, labels_train = train
features_test, labels_test = test


my_feature_columns = []
# print(train_x.keys())
print(features_train.keys())
for header in features_train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=header))
    print(tf.feature_column.numeric_column(key=header))
print(my_feature_columns)

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    model_dir='models/iris1',
    n_classes=3)
classifier.train(
    input_fn=lambda:train_input_fn(features_train, labels_train, batch_size=100),
    steps=1000)

eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(features_test, labels_test, batch_size=100))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
