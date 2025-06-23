import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Reshape, 
                                     Bidirectional, LSTM, Dense, Activation, 
                                     BatchNormalization, Dropout, Add, 
                                     TimeDistributed)

st.title("Handwritten Character Recognition System")
st.text("Please input Text Image for Recognition")

# Model selection
model_type = st.selectbox("Select Model Architecture", ["CNN-LSTM", "CNN","TrOCR"])

def preprocess(img):
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    (h, w) = img.shape
    final_img = np.ones([64, 256]) * 255  # white background
    
    # Maintain aspect ratio
    if w > 256:
        ratio = 256 / w
        img = cv2.resize(img, (256, int(h * ratio)))
        h, w = img.shape
        
    if h > 64:
        ratio = 64 / h
        img = cv2.resize(img, (int(w * ratio), 64))
        h, w = img.shape
        
    # Center the image
    start_y = (64 - h) // 2
    start_x = (256 - w) // 2
    final_img[start_y:start_y+h, start_x:start_x+w] = img
    
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

alphabets = u"!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
num_of_characters = len(alphabets) + 1  # +1 for CTC blank

def num_to_label(num):
    return "".join(alphabets[ch] for ch in num if ch != -1 and ch < len(alphabets))

@st.cache_resource
def load_model(model_type):
    if model_type == "CNN-LSTM":
        # CRNN with LSTM model
        input_data = Input(shape=(256, 64, 1), name='input')

        inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

        inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
        inner = Dropout(0.3)(inner)

        inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
        inner = Dropout(0.3)(inner)

        # CNN to RNN
        inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)
        inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

        ## RNN
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)

        ## OUTPUT
        inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
        y_pred = Activation('softmax', name='softmax')(inner)

        model = Model(inputs=input_data, outputs=y_pred)
        model.load_weights('/model file/CNN_LSTM_best.keras')
    
    else:  # CNN with 1D Convolutions
        def residual_block(x, filters, kernel_size=(3, 3)):
            shortcut = x
            # First convolution
            x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            # Second convolution
            x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            # Adjust shortcut if needed
            if shortcut.shape[-1] != filters:
                shortcut = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(shortcut)
                shortcut = BatchNormalization()(shortcut)
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            return x

        # Input Layer
        input_data = Input(shape=(256, 64, 1), name='input')

        # --- Enhanced CNN Backbone with Residual Connections ---
        x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(input_data)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)  # 128x32x32

        x = residual_block(x, 64)  # Residual block
        x = MaxPooling2D(pool_size=(2, 2))(x)  # 64x16x64
        x = Dropout(0.3)(x)

        x = residual_block(x, 128)  # Residual block
        x = MaxPooling2D(pool_size=(1, 2))(x)  # 64x8x128
        x = Dropout(0.4)(x)

        x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1, 2))(x)  # 64x4x256
        x = Dropout(0.4)(x)

        # --- Replace LSTM with TimeDistributed Dense Layers ---
        x = Reshape(target_shape=(64, 4*256))(x)  # (timesteps, 1024 features)

        # Apply dense layers with TimeDistributed
        x = TimeDistributed(Dense(512, activation='relu'))(x)
        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(512, activation='relu'))(x)
        x = Dropout(0.5)(x)

        # --- Output Layer ---
        x = TimeDistributed(Dense(num_of_characters))(x)
        y_pred = Activation('softmax', name='softmax')(x)

        model = Model(inputs=input_data, outputs=y_pred)
        model.load_weights('/model file/CNN_best.keras')
    
    return model

model = load_model(model_type)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and process image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Error: Could not read image. Please try another file.")
    else:
        # Convert BGR to RGB for correct color display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption='Original Image', use_container_width=True)  # Fixed parameter
        
        processed_img = preprocess(img)
        processed_img = processed_img / 255.0
        
        # Display model architecture
        st.subheader(f"Using Model: {model_type}")
        
        # Prediction
        with st.spinner('Recognizing text...'):
            pred = model.predict(processed_img.reshape(1, 256, 64, 1))
            
            # Calculate input length for CTC (time steps)
            input_length = np.ones(pred.shape[0]) * pred.shape[1]
            
            decoded = K.get_value(K.ctc_decode(
                pred, 
                input_length=input_length,
                greedy=True,
                beam_width=10,
                top_paths=1
            )[0][0])
            
            result = num_to_label(decoded[0])
            st.success(f"Recognized Text: {result}")
