tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)


images = data['image'].unique().tolist()
nimages = len(images)


split_index = round(0.85*nimages)
train_images = images[:split_index]
val_images = images[split_index:]


train = data[data['image'].isin(train_images)]
test = data[data['image'].isin(val_images)]


train.reset_index(inplace=True,drop=True)
test.reset_index(inplace=True,drop=True)


tokenizer.texts_to_sequences([captions[1]])[0]
