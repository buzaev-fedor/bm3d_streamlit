import streamlit as st
from PIL import Image
import numpy as np
from utils import add_gaussian_noise, array_to_image, compute_rmse_psnr
from bm3d import run_bm3d
import cv2

if __name__ == "__main__":

    st.header("Denoising images using BM3D")
    st.write("Choose any image and get corresponding denoising image")

    uploaded_file = st.file_uploader("Choose an image...")

    st.sidebar.title("Note")
    st.sidebar.write(
        """This playground was created for demonstrate BM3D.
           You can change parameters and see changes
        """
    )

    st.sidebar.title("Parameters for BM3D")
    sigma = st.sidebar.slider("sigma", 0, 100, 30, 5)
    tau_hard = 2500 if sigma <= 40 else 5000
    tau_wien = 400 if sigma <= 40 else 3500
    st.write(f"Tau hard = {tau_hard}")
    st.write(f"Tau Wiener = {tau_wien}")

    lambda_3D_hard = st.sidebar.slider("Threshold for Hard Thresholding", 0, 100, 27, 1) / 10

    n_hard = n_wien = st.sidebar.slider("n_hard, n_wien – search window size", 0, 100, 16, 1)

    k_hard = k_wien = st.sidebar.slider("k_hard, k_wien – size of patches", 0, 100, 8, 1)
    p_hard = p_wien = st.sidebar.slider("p_hard, p_wien - the loop over the pixels of the image "
                                        "is done with a step p (integer) in row and column.", 0, 10, 3, 1)
    useSD_hard = False
    tau_2D_hard = 'BIOR'

    useSD_wien = True
    tau_2D_wien = 'DCT'

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_size = st.number_input(f"image_size", min_value=0, max_value=2048, value=0, step=128)
        if image_size != 0:
            image = image.resize((image_size, image_size))
            st.image(image, caption='Input Image', use_column_width=True)
            if len(np.array(image).shape) != 2:
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            noised_image = add_gaussian_noise(np.array(image), sigma)
            st.image(noised_image, caption=f'Noised image with sigma={sigma}', use_column_width=True)
            first_step_image, denoised_image = run_bm3d(noised_image, sigma,
                                                        n_hard, k_hard, p_hard, tau_hard, useSD_hard, tau_2D_hard,
                                                        lambda_3D_hard,
                                                        n_wien, k_wien, p_wien, tau_wien, useSD_wien, tau_2D_wien)

            st.image(array_to_image(first_step_image), caption=f'image after first step with sigma={sigma}',
                     use_column_width=True)
            st.image(array_to_image(denoised_image), caption=f'Deoised image with sigma={sigma}', use_column_width=True)

            RMSE, PSNR = compute_rmse_psnr(noised_image, denoised_image)

            st.write(f"RMSE: {round(RMSE, 5)}")
            st.write(f"PSNR: {round(PSNR, 5)}")
