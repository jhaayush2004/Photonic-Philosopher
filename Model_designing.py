input1 = Input(shape=(1920,))
input2 = Input(shape=(max_length,))

img_features = Dense(256, activation='relu')(input1)
img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)


sentence_features = Embedding(vocab_size, 256, mask_zero=False)(input2)
merged = concatenate([img_features_reshaped,sentence_features],axis=1)


sentence_features = LSTM(350, return_sequences=True)(merged)

sentence_features= Dropout(0.2)(sentence_features)

sentence_features = LSTM(256, return_sequences=True)(sentence_features)

sentence_features = add([sentence_features, merged])  # Skip connection

sentence_features = LSTM(256)(sentence_features)

x = Dropout(0.3)(sentence_features)

x = add([x, img_features])

x = Dense(128, activation='relu')(x)

x = Dropout(0.5)(x)

output = Dense(vocab_size, activation='softmax')(x)


caption_model = Model(inputs=[input1,input2], outputs=output)
caption_model.compile(loss='categorical_crossentropy',optimizer='adam')
