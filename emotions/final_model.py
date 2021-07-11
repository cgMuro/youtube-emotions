import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from keras.utils.np_utils import to_categorical
import nlpaug.augmenter.word as naw, nac


# Define matplotlib style
plt.style.use('ggplot')



# ----------------------------------- Load and preprocess data ----------------------------------- #
def get_data():
    # Get dataset
    df = pd.read_csv('./data/labeled/youtube_labeled_edited.csv', usecols=['text', 'emotion'])

    # Extract features and labels
    x = df['text']
    y = df['emotion']

    # Get number of output classes
    N_EMOTIONS = len(df['emotion'].unique())

    # Encode categorical labels
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    y = to_categorical(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.05
    )

    return X_train, X_test, y_train, y_test, N_EMOTIONS



# ----------------------------------- Text augmentation ----------------------------------- #
class TextAugmentation():
    def __init__(self) -> None:
        pass

    def keyboard_augmenter(self, text):
        """ Substitute word by spelling mistake words dictionary. """
        aug = nac.KeyboardAug()
        keyboard_aug = aug.augment(text)
        return keyboard_aug

    def spelling_augmenter(self, text):
        """ Substitute word by spelling mistake words dictionary """
        aug = naw.SpellingAug()
        spelling_aug = aug.augment(text, n=2)
        return spelling_aug

    def augment_data(self, X, labels):
        """ Augments the given data. """
        X_augmented = np.array([])
        augmented_labels = np.array([])

        augmenters = [
            self.keyboard_augmenter,
            self.spelling_augmenter
        ]

        for idx, example in enumerate(X):
            for augmenter in augmenters:
                augmented_example = augmenter(example)
                augmented_example = np.array([*augmented_example]) if isinstance(augmented_example, list) else np.array([augmented_example])
                X_augmented = np.concatenate((X_augmented, augmented_example))
                for _ in range(len(augmented_example)):
                    if len(augmented_labels) == 0:
                        augmented_labels = np.concatenate((augmented_labels, labels[idx]))
                    else:
                        augmented_labels = np.vstack((augmented_labels, labels[idx]))

        X = np.concatenate((X, X_augmented))
        print('Data after augmentation:', len(X))

        labels = np.concatenate((labels, augmented_labels))
        print('Labels data after augmentation:',len(labels))

        # Count augmented data by class
        count = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        }

        for i in labels:
            count[tf.math.argmax(i).numpy()] += 1

        print('-'*34)
        print(' constructive feedback/idea |', count[0])
        print(' negative', ' '*17 ,  '|', count[1])
        print(' neutral/other', ' '*12 ,  '|', count[2])
        print(' positive', ' '*17 ,  '|', count[3])
        print(' sadness', ' '*18 ,  '|', count[4])
        print('-'*34)
        print('\n\n')

        return np.asarray(X), np.asarray(labels)



# ----------------------------------- Model ----------------------------------- #
def get_model(output_classes: int, print_summary: bool):
    """ Build and return the model. """
    # Path to model in TensorFlow Hub
    model_hub_path = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"

    # Build first module layers using TensorFlow Hub model
    hub_layer = hub.KerasLayer(model_hub_path, input_shape=[], dtype=tf.string, trainable=False)

    # Build sequential model
    model = tf.keras.models.Sequential([
        hub_layer,
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_classes, activation='sigmoid')
    ])

    # Define loss, optimizer and metrics
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy', tf.keras.metrics.AUC(), tfa.metrics.F1Score(output_classes), tfa.metrics.CohenKappa(output_classes)]
    )

    # Print model structure if requested
    if print_summary:
        print('\n')
        print(model.summary())
        print('\n')

    return model



# ----------------------------------- Cyclical Learning ----------------------------------- #
# https://github.com/bckenstler/CLR
class CyclicLR(tf.keras.callbacks.Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())



# ----------------------------------- Training and testing ----------------------------------- #

def train_model(model, epochs: int = 1500, batch_size: int = 3000):
    """ Train give model with cyclical learning. """
    clr = CyclicLR(base_lr=0.0003, max_lr=0.003, step_size=2000., mode='triangular2')

    results = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_split=0.05,
        shuffle=True,
        batch_size=batch_size,
        callbacks=[clr]
    )

    return results

def test_model(model):
    """ Test given model. """
    res = model.evaluate(X_test, y_test)
    names = model.metrics_names
    for i in range(len(res)):
        print(f'{names[i].upper()}:   {res[i]}')


def save_model(model, path: str):
    """ Save given model. """
    print('\nSaving model...')
    tf.saved_model.save(model, path)
    print('Model saved.\n')



# ----------------------------------- New predictions ----------------------------------- #
def predict(model, sentence: str, print_values: bool):
    """ Use given model for new predictions. """
    prediction = model.predict(np.array([sentence]))

    if print_values:
        print(prediction)

    return decode_map[np.argmax(prediction)]




if __name__ == '__main__':
    # Get data split and number of output classes
    X_train, X_test, y_train, y_test, N_EMOTIONS = get_data()

    # Define decode map
    decode_map = {
        0: 'constructive feedback/idea',
        1: 'negative',
        2: 'neutral/other', 
        3: 'positive', 
        4: 'sadness', 
    }

    # Init text agumenter
    text_augmenter = TextAugmentation()

    # Augment text
    X_train, y_train = text_augmenter.augment_data(X_train, y_train)
    X_test, y_test = text_augmenter.augment_data(X_test, y_test)

    # Shuffle augmented data
    X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
    X_test, y_test = sklearn.utils.shuffle(X_test, y_test)

    print('\n\n')
    print('X_train:', len(X_train), '|', 'X_test:', len(X_test))
    print('y_train:', len(y_train), '|', 'y_test:', len(y_test))
    print('\n\n')

    # Init model
    model = get_model(output_classes=N_EMOTIONS, print_summary=True)

    # Train model
    results = train_model(model, epochs=1500, batch_size=3000)

    # Test model
    test_model(model)

    # Save model
    save_model(model, path='')

    # New predictions
    output = predict(model, sentence='', print_values=True)

    print('\n', output)
