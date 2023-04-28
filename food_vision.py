import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
#from tensorflow_helper import multiclass_pred_new_image_plotly, load_and_prep_image
import pandas as pd
import os
import gdown


st.set_page_config(layout="wide")

food_list = ['curry de pollo', 'alitas de pollo', 'arroz frito', 'salmón a la parrilla', 'hamburguesa', 'helado',
             'pizza', 'ramen', 'bife', 'Sushi']


# Function to load and prepare the image
def load_and_prepare_image(img, rescale=False):
    img = Image.open(img).convert('RGB')
    img = img.resize((224, 224))  # Resize the image to required dimensions
    if rescale:
        img = np.array(img) / 255.0  # Normalize the pixel values
    return np.expand_dims(img, axis=0)  # Add batch dimension


# Function to load the model
@st.cache_resource()
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

@st.cache_resource()
def download_model_from_google_drive(url, output_name):
    gdown.download(url, output_name, quiet=False)
    # Now load the model from the file
    model = load_model(output_name)
    return model



@st.cache_resource()
def load_pickle(path):
    with open(f'{path}', 'rb') as file:
        le_loaded = pickle.load(file)
        return le_loaded

@st.cache_resource
def load_numpy_array(path):
    embedded_features = np.load(path)
    return embedded_features


# Function to make predictions
def predict(image, model):
    prepared_image = load_and_prepare_image(image)
    return model.predict(prepared_image)


# Define the Streamlit app
def main():
    label_encoder = load_pickle("label_encoder.pkl")
    #embedded_features = load_numpy_array("features.npy")
    st.markdown("<h1 style='text-align: center; color: red;'>Food Vision!, Classification App</h1>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col2.info("**Esta aplicacion es capaz de clasificar imagenes de las siguientes 10 clases de comida:**")
    #col2.info(f"**<font size='5'>Esta aplicacion es capaz de clasificar imagenes de las siguientes clases:</font>**",
   #            unsafe_allow_html=True)
    index = pd.RangeIndex(start=1, stop=11)

    data = list(zip(label_encoder.classes_, food_list))
    df = pd.DataFrame(data=data, columns=["Food Classes", "Clases de Comida"],
                      index=index).transpose()
    col2.dataframe(df, use_container_width=True)

    with col1:
        col1.image("midjourney/FsO4D-3XoAAUrEQ.jpeg")
    #
    #     col1_left, col1_right = st.columns(2)
    #
    #     for idx, n in enumerate(os.listdir("images_to_test")):
    #         if idx <= 4:
    #             col1_left.image(f"images_to_test/{n}", use_column_width=True)
    #         else:
    #             col1_right.image(f"images_to_test/{n}", use_column_width=True)
    #
    with col3:
        col3.image("midjourney/d.jpeg")

        pass
    #
    #     col3_left, col3_right = st.columns(2)
    #
    #     for idx, n in enumerate(os.listdir("images_to_test")):
    #         if idx > 4:
    #             col3_left.image(f"images_to_test/{n}", use_column_width=True)
    #         else:
    #             col3_right.image(f"images_to_test/{n}", use_column_width=True)

    # Upload the image
    image = col2.file_uploader("**Subir Imagen para Clasificar.**", type=["jpg", "jpeg", "png"])

    # Load the pre-trained model (change the path to your model)
    #model_path = "food_vis_model.h5"
    #model = load_model(model_path)

    file_id = "1jJIhQRTHUh5cKF9SZVyNT0FyGAVnQ-mf"
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    model = download_model_from_google_drive(url, "model.h5")
    #embedding_model = load_model("embedding_model.h5")
    #outlier_detector = load_encoder("outlier_detector.pkl")
    #embedded_features = load_numpy_array("features.npy")

    # Make a prediction and display the result
    if image is not None:
        #st.image(image, use_column_width=True)
        im1 = load_and_prepare_image(image, rescale=False)
        #new_feature_vector = embedding_model.predict(im1)
        prediction = model.predict(im1)
        # prediction = predict(image, model)
        predicted_class_encoded = np.argmax(prediction)
        predicted_class = label_encoder.inverse_transform([predicted_class_encoded])

        #col2.write(f"{predicted_class[0]}")

        #CALCAULATE Mahalanobis Distance-based Confidence Score (MCD)
        # mean = np.mean(embedded_features, axis=0)
        # covariance = np.cov(embedded_features, rowvar=False)
        # inv_covariance = np.linalg.inv(covariance + 1e-6 * np.eye(covariance.shape[0]))
        #
        # mahalanobis_distance = np.sqrt(np.dot(np.dot((new_feature_vector - mean), inv_covariance), (new_feature_vector - mean).T).diagonal())
        # #anomaly_score = outlier_detector.decision_function(new_feature_vector.reshape(1, -1))
        # # threshold = np.percentile(anomaly_score, 50)
        # # is_outlier = outlier_detector.predict(new_feature_vector.reshape(1, -1))
        # if mahalanobis_distance[0] > 70:
        #     col2.info("**Mahalanobis rechaza esta imagen al ser muy distinta a las imagenes con las que fue entranado el modelo.**")
        #     col2.write(f"mahalanobis_distance: {round(mahalanobis_distance[0],2)}")
        #
        # else:

        confidence = prediction[0][predicted_class_encoded]

        if confidence < 0.75:
            col2.info("**El modelo no esta seguro de su prediccion. Confianza inferior a 75%.**")

        else:
            #col2.write(f"{confidence}")
            df_t = df.transpose()
            pred_class = df_t[df_t["Food Classes"] == f"{predicted_class[0]}"]["Clases de Comida"].to_list()[0]

            # Display the predicted class
            col2.write(f"<font size='6'>El modelo piensa que esto es **<font color='red'>{pred_class}!</font>**</font>", unsafe_allow_html=True)


            col2.image(image, use_column_width=True)
            # fig2 = multiclass_pred_new_image_plotly(model, image, label_encoder.classes_)
            # col2.plotly_chart(fig2, use_container_width=True)
    if image is None:
        imdefault = load_and_prepare_image("helado.jpg", rescale=False)
        prediction = model.predict(imdefault)
        predicted_class = np.argmax(prediction)
        predicted_class = label_encoder.inverse_transform([predicted_class])

        df_t = df.transpose()
        pred_class = df_t[df_t["Food Classes"] == f"{predicted_class[0]}"]["Clases de Comida"].to_list()[0]

        # Display the predicted class
        col2.write(f"<font size='6'>El modelo piensa que esto es **<font color='red'>{pred_class}!</font>**</font>", unsafe_allow_html=True)

        col2.image("helado.jpg", use_column_width=True)



if __name__ == "__main__":
    main()

# The line if __name__ == "__main__": is a common Python idiom used to check if the current script is being run as
# the main program (i.e., not being imported as a module into another script). If the script is being run as the main
# program, the condition is true, and the code within the block will be executed. In this case, it calls the main()
# function.

# When a Python script is executed, a special variable called __name__ is created. If the script is being run as the
# main program, the value of __name__ is set to "__main__". If the script is being imported as a module into another
# script, the value of __name__ will be the name of the module.

# In the context of the given Streamlit app code, this line ensures that the main() function is only called when the
# script is run directly. If the script were to be imported into another script as a module, the main() function
# would not be called automatically upon import.
