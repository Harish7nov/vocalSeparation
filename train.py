import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.layers import *
import time
tf.compat.v1.disable_eager_execution()
import soundfile as sf
import pandas as pd
import os
loc = os.getcwd()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
config = tf.compat.v1.ConfigProto()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.compat.v1.Session(config=config)
os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '256'


class Generator(tf.keras.utils.Sequence):
    def __init__(self, inp_files, out_files, batch_size, steps, n_songs):
        self.batch_size = batch_size
        self.inp_files = inp_files
        self.out_files = out_files
        self.steps = steps
        self.n_songs = n_songs

        # Set the frame length
        self.n_fft = 4096
        # Set the hop length
        self.hop_length = self.n_fft // 4
        # Set the sampling Frequency
        self.sr = 44100
        # Define the number of time frames needed
        self.time = 256
        # Calculate the duration of 1 STFT frame in milliseconds
        self.dur = self.hop_length / self.sr
        # Calculate the duration for the audio length that
        # needs to be cropped
        self.seconds = np.round(self.time * self.dur, 4)

        # Output STFT feature shape would be 
        # Time Steps x Frequency Bins x n_channels
        
    def __len__(self):
        return self.steps

    def audio_scale(self, audio, amplitude):
        scale_factor = amplitude * np.sqrt(np.mean(audio * audio))
        scaled_audio = audio * scale_factor

        return scaled_audio

    def read_data(self, x):
        inp = []
        out = []
        idx = np.zeros(self.batch_size)
        idx[self.batch_size // 2:] = 1
        np.random.shuffle(idx)
        amp_idx = np.arange(0.4, 1.3, 0.05)
        amp_idx = np.random.choice(amp_idx, size=self.batch_size, replace=False)

        for i in range(self.batch_size):
            
            # Read the input mixture audio
            audio, _ = sf.read(self.inp_files[x[i]])
            # convert to mono audio
            audio = np.mean(audio, axis=1)
            duration = len(audio) / self.sr
            # Read the particular duration audio
            index = np.random.choice(list(range(int(duration - self.seconds))), size=1)[0]
            temp = audio[index * self.sr : int((index + self.seconds) * self.sr)]

            if idx[i]:
                temp = self.audio_scale(temp, amp_idx[i])
            # Calculate the STFT features on the go
            feat = np.abs(librosa.stft(temp, self.n_fft, self.hop_length).T)[:, :-1]
            # feat = librosa.amplitude_to_db(feat, ref=np.max)
            inp.append(feat)
            
            # Read the source vocal audio as the ground truth
            audio, _ = sf.read(self.out_files[x[i]])
            # convert to mono audio
            audio = np.mean(audio, axis=1)
            temp = audio[index * self.sr : int((index + self.seconds) * self.sr)]

            if idx[i]:
                temp = self.audio_scale(temp, amp_idx[i])
            # Calculate the STFT features on the go
            feat = np.abs(librosa.stft(temp, self.n_fft, self.hop_length).T)[:, :-1]
            # feat = librosa.amplitude_to_db(feat, ref=np.max)
            out.append(feat)

        inp = np.expand_dims(np.array(inp), axis=-1)
        out = np.expand_dims(np.array(out), axis=-1)
        
        return inp, out
        
    def __getitem__(self, idx):

        idx = np.random.choice(list(range(self.n_songs)), size=self.batch_size, replace=False)
        inp, out = self.read_data(idx)
        
        return inp, out


def get_model():

    stride = 2
    use_bias = True
    input_layer = Input(shape = [256, 2048, 1])

    block1_conv = Conv2D(16, (5, 5), padding='same', strides=stride, 
                                use_bias=use_bias)(input_layer)
    block1_act = LeakyReLU()(block1_conv)
    block1_batch = BatchNormalization()(block1_act)
    # block1_pool = MaxPooling2D((2, 2), strides=(2, 2))(block1_batch)

    block2_conv = Conv2D(32, (5, 5), padding='same', strides=stride, 
                                use_bias=use_bias)(block1_batch)
    block2_act = LeakyReLU()(block2_conv)
    block2_batch = BatchNormalization()(block2_act)
    # block2_pool = MaxPooling2D((2, 2), strides=(2, 2))(block2_batch)

    block3_conv = Conv2D(64, (5, 5), padding='same', strides=stride, 
                                use_bias=use_bias)(block2_batch)
    block3_act = LeakyReLU()(block3_conv)
    block3_batch = BatchNormalization()(block3_act)
    # block3_pool = MaxPooling2D((2, 2), strides=(2, 2))(block3_batch)

    block4_conv = Conv2D(128, (5, 5), padding='same', strides=stride, 
                                use_bias=use_bias)(block3_batch)
    block4_act = LeakyReLU()(block4_conv)
    block4_batch = BatchNormalization()(block4_act)
    # block4_pool = MaxPooling2D((2, 2), strides=(2, 2))(block4_batch)

    block5_conv = Conv2D(256, (5, 5), padding='same', strides=stride, 
                                use_bias=use_bias)(block4_batch)
    block5_act = LeakyReLU()(block5_conv)
    block5_batch = BatchNormalization()(block5_act)
    # block5_pool = MaxPooling2D((2, 2), strides=(2, 2))(block5_batch)

    block6_conv = Conv2D(512, (5, 5), padding='same', strides=stride, 
                                use_bias=use_bias)(block5_batch)
    block6_act = LeakyReLU()(block6_conv)
    block6_batch = BatchNormalization()(block6_act)
    # block6_pool = MaxPooling2D((2, 2), strides=(2, 2))(block6_batch)

    # End of Encoder

    # Start of Decoder

    model = Conv2DTranspose(256, (5, 5), (2, 2), padding='same', use_bias=use_bias)(block6_batch)
    model = ReLU()(model)
    model = BatchNormalization()(model)
    model = SpatialDropout2D(0.4)(model)


    model = Concatenate()([block5_batch, model])

    model = Conv2DTranspose(128, (5, 5), (2, 2), padding='same', use_bias=use_bias)(model)
    model = ReLU()(model)
    model = BatchNormalization()(model)
    model = SpatialDropout2D(0.4)(model)

    model = Concatenate()([block4_batch, model])

    model = Conv2DTranspose(64, (5, 5), (2, 2), padding='same', use_bias=use_bias)(model)
    model = ReLU()(model)
    model = BatchNormalization()(model)
    model = SpatialDropout2D(0.4)(model)

    model = Concatenate()([block3_batch, model])

    model = Conv2DTranspose(32, (5, 5), (2, 2), padding='same', use_bias=use_bias)(model)
    model = ReLU()(model)
    model = BatchNormalization()(model)

    model = Concatenate()([block2_batch, model])
            
    model = Conv2DTranspose(16, (5, 5), (2, 2), padding='same', use_bias=use_bias)(model)
    model = ReLU()(model)
    model = BatchNormalization()(model)

    model = Concatenate()([block1_batch, model])

    model = Conv2DTranspose(1, (5, 5), (2, 2), padding='same', use_bias=use_bias)(model)
    model = ReLU()(model)
    model = BatchNormalization()(model)

    # model = Concatenate()([block1, model])

    model = Conv2D(1, (5, 5), dilation_rate=2, activation='sigmoid', 
                            padding='same')(model)

    out = Multiply()([model, input_layer])
    model = tf.keras.models.Model(input_layer, out)
    
    return model


if __name__ == "__main__":

    learning_rate = 5e-04
    batch_size = 4
    epochs = 500
    train_steps = 500
    valid_steps = 100

    # Define the number of songs
    # in train set and test set
    train_songs = 100
    valid_songs = 50

    train_path = os.path.join(loc, r"MUSDB 18\train")
    test_path = os.path.join(loc, r"MUSDB 18\test")

    # train_path = r"D:\Downloads\MUSDB 18\train"
    # test_path = r"D:\Downloads\MUSDB 18\test"

    train_inp_filenames = []
    train_out_filenames = []

    test_inp_filenames = []
    test_out_filenames = []

    for i in sorted(os.listdir(train_path)):
        path = os.path.join(train_path, i)

        train_inp_filenames.append(os.path.join(path, "mixture.wav"))
        train_out_filenames.append(os.path.join(path, "vocals.wav"))


    for i in sorted(os.listdir(test_path)):
        path = os.path.join(test_path, i)

        test_inp_filenames.append(os.path.join(path, "mixture.wav"))
        test_out_filenames.append(os.path.join(path, "vocals.wav"))


    train_gen = Generator(train_inp_filenames, train_out_filenames, batch_size, train_steps, train_songs)
    valid_gen = Generator(test_inp_filenames, test_out_filenames, batch_size, valid_steps, valid_songs)

    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    # opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.98, nesterov=True)
    # opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model = get_model()
    print(model.summary())
    model.compile(opt, loss="mean_absolute_error")

    # Define the keras call back for model checkpoint
    mcp1 = tf.keras.callbacks.ModelCheckpoint(r"valid_model.h5", monitor='val_loss', save_best_only=True,
                            mode='min')

    mcp2 = tf.keras.callbacks.ModelCheckpoint(r"train_model.h5", monitor='loss', save_best_only=True,
                            mode='min')

    log_dir = f'logs\Vocal Separation - {time.strftime("%H-%M-%S", time.localtime())}'
    tensorboard = tf.compat.v1.keras.callbacks.TensorBoard(log_dir=log_dir, write_grads=True)

    history = model.fit(train_gen, 
                        epochs=epochs,
                        validation_data=valid_gen,
                        steps_per_epoch=train_steps,
                        validation_steps=valid_steps,
                        shuffle=True,
                        workers=3,
                        callbacks=[mcp1, mcp2, tensorboard]
                    )

    hist_df = pd.DataFrame(history.history)
    hist_csv_file = os.path.join(loc, r'history.csv')

    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
