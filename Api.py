import tensorflow as tf
from fastapi import FastAPI, UploadFile, HTTPException
from tensorflow.keras.models import load_model
import uvicorn
import numpy as np

# Load the saved model
try:
    cnn_model = load_model("BearCnn.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the CNN model. Ensure the file 'BearCnn.h5' is available.")

# Initialize FastAPI
app = FastAPI()


def preprocess_image_for_prediction(contents, img_size=(64, 64)):
    """
    Preprocess the uploaded image to match the input shape required by the CNN model.
    """
    try:
        # Decode image and preprocess
        img = tf.image.decode_image(contents, channels=3)
        img = tf.image.resize(img, img_size)
        img = img / 255.0  # Normalize pixel values to [0, 1]
        return tf.expand_dims(img, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        raise ValueError("Failed to preprocess the image. Ensure the file is a valid image format.")


@app.post("/predict/")
async def predict(image: UploadFile):
    """
    Predict whether the uploaded image is a bear or not.
    """
    try:
        # Read the contents of the file
        contents = await image.read()

        # If no contents are uploaded, raise an error
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # Preprocess the image for the model
        preprocessed_img = preprocess_image_for_prediction(contents)

        # Get prediction from the model
        prediction = cnn_model.predict(preprocessed_img)
        print(f"Model prediction output: {prediction}")

        # Interpreting the prediction (binary classification thresholding)
        predicted_value = prediction[0][0]  # Assuming binary classification output [[value]]
        label = "bear" if predicted_value > 0.5 else "not bear"

        return {"prediction": label, "confidence": float(predicted_value)}

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during prediction. Check server logs for details."
        )


# Run the server with the specified host and port (5000)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)


# import tensorflow as tf
# from fastapi import FastAPI, UploadFile, HTTPException
# from tensorflow.keras.models import load_model
# import uvicorn
# import numpy as np
#
# # Load the saved model
# try:
#     cnn_model = load_model("BearCnn.h5")
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     raise RuntimeError("Failed to load the CNN model. Ensure the file 'BearCnn.h5' is available.")
#
# # Initialize FastAPI
# app = FastAPI()
#
# def preprocess_image_for_prediction(contents, img_size=(64, 64)):
#     """
#     Preprocess the uploaded image to match the input shape required by the CNN model.
#     """
#     try:
#         img = tf.image.decode_image(contents, channels=3)
#         img = tf.image.resize(img, img_size)
#         img = img / 255.0  # Normalize pixel values
#         return tf.expand_dims(img, axis=0)  # Add batch dimension
#     except Exception as e:
#         print(f"Error in image preprocessing: {e}")
#         raise ValueError("Failed to preprocess the image. Ensure the file is a valid image format.")
#
# @app.post("/predict/")
# async def predict(file: UploadFile):
#     """
#     Predict whether the uploaded image is a bear or not.
#     """
#     try:
#         # Read and preprocess the uploaded file
#         contents = await file.read()
#         if not contents:
#             raise HTTPException(status_code=400, detail="Empty file uploaded.")
#
#         preprocessed_img = preprocess_image_for_prediction(contents)
#
#         # Get prediction from the CNN model
#         prediction = cnn_model.predict(preprocessed_img)
#         print(f"Model prediction output: {prediction}")
#
#         # Interpret the prediction
#         predicted_value = prediction[0][0]  # Assuming model output is [[value]]
#         label = "bear" if predicted_value > 0.5 else "not bear"
#
#
#         return {"prediction": label, "confidence": float(predicted_value)}
#
#     except HTTPException as http_err:
#         raise http_err
#
#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail="An error occurred during prediction. Check server logs for details."
#         )
#
# # Run the server
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
