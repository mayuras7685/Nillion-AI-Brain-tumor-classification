import streamlit as st
import os
import sys
import json
import random
import io
import contextlib
from dotenv import load_dotenv
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import asyncio
from common.utils import compute, store_secret_array
from nillion_python_helpers import create_nillion_client, create_payments_config
from py_nillion_client import NodeKey, UserKey
import py_nillion_client as nillion
import nada_numpy as na
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey

label = {"0": "Benign", "1": "Malignant"}

# Load environment variables
def load_environment():
    load_dotenv(f".env")
    
async def store_images(model_user_client, payments_wallet, payments_client, cluster_id, test_image_batch, secret_name, nada_type, ttl_days, permissions):
    images_store_id = await store_secret_array(
        model_user_client,
        payments_wallet,
        payments_client,
        cluster_id,
        test_image_batch,
        secret_name,
        nada_type,
        ttl_days,
        permissions,
    )
    return images_store_id

async def compute_results(model_user_client, payments_wallet, payments_client, program_id, cluster_id, compute_bindings, model_store_id, images_store_id):
    result = await compute(
        model_user_client,
        payments_wallet,
        payments_client,
        program_id,
        cluster_id,
        compute_bindings,
        [model_store_id, images_store_id],
        nillion.NadaValues({}),
        verbose=True,
    )
    return result

def softmax_two_values(x1, x2):
    x = np.array([x1, x2]) / 1_000_000  # Scale the inputs by dividing by one million
    e_x = np.exp(x - np.max(x))  # Compute the exponential of the inputs for numerical stability
    return e_x / e_x.sum()        # Normalize to get percentages

def main():
    st.title("Melanoma detection webapp - ETHGlobal Brussels")

    # Load environment variables
    load_environment()

    # Display some environment information
    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    grpc_endpoint = os.getenv("NILLION_NILCHAIN_GRPC")

    st.write(f"Cluster ID: {cluster_id}")
    st.write(f"gRPC Endpoint: {grpc_endpoint}")

    if not cluster_id or not grpc_endpoint:
        st.error("Environment variables for NILLION_CLUSTER_ID or NILLION_NILCHAIN_GRPC are not set.")
        return

    uploaded_file = st.file_uploader("Choose an image", type=["jpeg", "jpg", "png"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Transform and process the image
        transformed_image = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
        ])(image)

        model_user_userkey = UserKey.from_seed("abc")
        model_user_nodekey = NodeKey.from_seed(str(random.randint(0, 1000)))

        # Debugging: Print the keys
        st.write(f"UserKey: {model_user_userkey}")
        st.write(f"NodeKey: {model_user_nodekey}")

        # Create Nillion client
        try:
            model_user_client = create_nillion_client(model_user_userkey, model_user_nodekey)
        except Exception as e:
            st.error(f"Error creating Nillion client: {e}")
            return

        payments_config = create_payments_config(os.getenv("NILLION_NILCHAIN_CHAIN_ID"), grpc_endpoint)
        payments_client = LedgerClient(payments_config)
        payments_wallet = LocalWallet(
            PrivateKey(bytes.fromhex(os.getenv("NILLION_NILCHAIN_PRIVATE_KEY_0"))),
            prefix="nillion",
        )

        with open("src/data/tmp.json", "r") as provider_variables_file:
            provider_variables = json.load(provider_variables_file)

        program_id = provider_variables["program_id"]
        model_store_id = provider_variables["model_store_id"]
        model_provider_party_id = provider_variables["model_provider_party_id"]

        test_image_batch = np.array(transformed_image.unsqueeze(0))
        permissions = nillion.Permissions.default_for_user(model_user_client.user_id)
        permissions.add_compute_permissions({model_user_client.user_id: {program_id}})

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            try:
                images_store_id = asyncio.run(store_images(model_user_client, payments_wallet, payments_client, cluster_id, test_image_batch, "my_input", na.SecretRational, 1, permissions))
                print(f"Images Store ID: {images_store_id}")

                compute_bindings = nillion.ProgramBindings(program_id)
                compute_bindings.add_input_party("Provider", model_provider_party_id)
                compute_bindings.add_input_party("User", model_user_client.party_id)
                compute_bindings.add_output_party("User", model_user_client.party_id)

                result = asyncio.run(compute_results(model_user_client, payments_wallet, payments_client, program_id, cluster_id, compute_bindings, model_store_id, images_store_id))
                first_key = next(iter(result))
                parts = first_key.split('_')
                number_part = parts[-1]
                st.write(f"Predicted class: {label[str(number_part)]}")
                print(f"Predicted class: {label[str(number_part)]}")
            except Exception as e:
                st.error(f"Error during computation: {e}")

        st.markdown("### Terminal Output")
        st.text_area("Output", output.getvalue(), height=300)

if __name__ == "__main__":
    main()
