from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, GlobalMaxPooling2D
from tensorflow.keras.models import load_model, Model
import numpy as np
import sys

message = input("""This program is classify cats and dogs images, 
    Please put your image in the image directory and the program will Tell you !,
     Are you sure : Y or N:\n""")
if message == "Y":
    print("Ok lets Do it !!!! ")
else :
    sys.exit()



datagen = ImageDataGenerator(rescale=1./255 ,preprocessing_function=preprocess_input)

test_datagen = datagen.flow_from_directory('image',
                                            batch_size=1,
                                            shuffle=False,
                                          )
if len(test_datagen) > 1:
    print("Error More Than One Image")
    sys.exit()

model = VGG16(weights='imagenet', include_top=False)

inp = Input(shape=(224, 224, 3), batch_size=1)

x = model(inp)
x = GlobalMaxPooling2D()(x)

model_feature_extractor = Model(inputs=[inp], outputs=[x])
X_test = np.zeros((1, 512), dtype=np.float32)
X_test = model_feature_extractor.predict(test_datagen)


new_model =load_model('DogVsCat/my_model')


preds = new_model.predict(X_test)

if preds[0] >= 0.5:
    print("The Image Contains a Dog")
else:
    print("The Image Contains a Cat")                                            

