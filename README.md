# Brain Tumor classification
Classifying brain tumor images using a pre-trained deep learning model. The app allows users to upload MRI images and predicts the type of brain tumor present in the image.

## Dataset

The model is trained on the [Brain Tumor MRI Dataset]([https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)) which consists of 13,900 high-resolution images of both benign and malign tumors.

## Model

The neural network was developed using `nada_ai` from Nillion and was trained to meet size constraints while maintaining robust performance metrics:

|                | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Glioma         | 0.99      | 0.97   | 0.98     | 300     |
| Meningioma     | 0.97      | 0.97   | 0.97     | 306     |
| No Tumor       | 0.99      | 1.00   | 0.99     | 405     |
| Pituitary      | 0.99      | 1.00   | 0.99     | 300     |
| **Accuracy**   |           |        | 0.99     | 1311    |
| **Macro Avg**  | 0.99      | 0.98   | 0.98     | 1311    |
| **Weighted Avg** | 0.99    | 0.99   | 0.99     | 1311    |

**Test Accuracy**: 98.55%

In order to host the model on Nillion one must run the provider.ipynb notebook.

# yet not completed

## Hosting

The model is hosted on the Nillion testnet, and we employ a Streamlit webapp allowing users to upload a brain tumor image and quickly receive a prediction. The service is currently free, with transaction costs on the testnet covered by our wallet.

## Usage

To use the prediction service, please follow the isntruction on the next step:

### Inference 

1. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate 
```

2. Install all the required library by:
```
 pip install -r requirements.txt
 ```
3. Create an .env file according to `https://docs.nillion.com/network-configuration` and place it inside `/nillion/quickstart/nada_quickstart_programs`
4. Run the streamlit platfor with:
 ```
 Streamlit run main.py
 ```
5. Upload an image of a MRI
6. Receive your prediction.

The image is processed privately without being stored on any server, ensuring user data remains confidential.
