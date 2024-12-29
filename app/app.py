import os
import sys
import base64
from PIL import Image
from io import BytesIO
import boto3
import time

# Add video_analytics_lib to the path immediately
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Initialize Streamlit
import streamlit as st
print("Boto3 version: ", boto3.__version__)
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Add the parent directory (project-folder/) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#Tools
import tools.image_background_lib.image_background_lib as bg_glib
import tools.image_understanding_lib.image_understanding_lib as ui_glib
import tools.image_extension_lib.image_extension_lib as img_extension_glib
import tools.subtitle_translation_lib.subtitle_translation_lib as sub_tran_glib
import tools.image_generation_lib.image_generation_lib as img_gen_glib
import tools.object_replace_remove_lib.object_replace_remove_lib as rplce_rmv_glib
import tools.video_analytics_lib.video_analytics_lib as vid_analytics_lib
import tools.image_to_video_lib.image_to_video_lib as img_to_video_glib
import tools.text_to_video_lib.text_to_video_lib as txt_to_video_glib
import tools.background_removal_lib.background_removal_lib as bg_removal_glib
import tools.image_variation_lib.image_variation_lib as img_variation_glib
import tools.image_conditioning_lib.image_conditioning_lib as img_condition_glib
import tools.color_guided_lib.color_guided_lib as clr_guided_glib

# S3 Client initialization
bucket_name = "demo-portal-videos-jossai-east1"

# Global variable for video_compression function
vid_analytics_lib.video_compression = None


# Load custom CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="Image App")

# Initialize the session state for resized images and super-resolution images
if "resized_image_io" not in st.session_state:
    st.session_state["resized_image_io"] = None
if "high_res_image_io" not in st.session_state:
    st.session_state["high_res_image_io"] = None

load_css()

# Create tabs: Multimodal & Images, Video Subtitle/Translation, Image to Video Demo, and Video Analytics
tab1, tab2, tab3, tab4 = st.tabs(["Content/Image Generation", "Video Subtitle/Translation", "Text/Image to Video", "Video Analytics"])


# Tab 1: Multimodal & Images (containing subtabs)
with tab1:
    st.title("Image/Text Generation")
    subtab1, subtab2, subtab3, subtab4, subtab5, subtab6, subtab7, subtab8, subtab9 = st.tabs(["Image Generation", "Image Understanding", "Background Change(Outpainting)", "Image Extension(Outpainting)", "Image Replace/Remove", "Image Variation", "Image Conditioning", "Color Guided Content", "Background Removal"])



    # Subtab: Image Generation
    with subtab1:
        st.title("Image Generation")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Image parameters")
            # Add the model selection widget
            # Update the model selection options
            model_options = [ 
                "Nova Canvas",
                "Amazon Titan V1", 
                "Amazon Titan V2", 
                "Stable Diffusion 3 Large", 
                "Stable Image Ultra", 
                "Stable Image Core",
                "Custom Amazon Titan Model V2",
                "Stable Diffusion"
            ]            
            selected_model = st.selectbox("Select Model", model_options)
            prompt_text = st.text_area("What you want to see in the image:", height=100, value="L'Oreal store that appears busy with alot of people shopping.")
            negative_prompt = st.text_input("What should not be in the image:")
            
            # Mapping the user-friendly model names to actual Bedrock model IDs
            model_id_mapping = {
                "Nova Canvas": "amazon.nova-canvas-v1:0",
                "Amazon Titan V1": "amazon.titan-image-generator-v1",
                "Amazon Titan V2": "amazon.titan-image-generator-v2:0",
                "Stable Diffusion 3 Large": "stability.sd3-large-v1:0",
                "Stable Image Ultra": "stability.stable-image-ultra-v1:0",
                "Stable Image Core": "stability.stable-image-core-v1:0",
                "Stable Diffusion": "stability.stable-diffusion-xl-v1",
                "Custom Amazon Titan Model V2": "xxx" # Replace with provisioned thoughput ARN for the fine tuned bedrock model

            }

            generate_button = st.button("Generate", type="primary", key="gen_generate_button_image_generation")

        with col2:
            st.subheader("Result")
            if generate_button:
                with st.spinner("Drawing..."):
                    model_id = model_id_mapping[selected_model]  # Get the actual model ID from the mapping
                    generated_image = img_gen_glib.get_image_from_model(
                        prompt_content=prompt_text, 
                        negative_prompt=negative_prompt, 
                        model_id=model_id
                    )
                st.image(generated_image)









    # Subtab: Image Understanding
    with subtab2:
        st.title("Image Understanding")

        col1, col2, col3 = st.columns(3)

        prompt_options_dict = {
            "Product Description": "Write a compelling and detailed product description for this L'Oreal cosmetic product. Highlight its key features, benefits, and ingredients. Focus on conveying a sense of luxury, innovation, and the brand's commitment to beauty and skincare. Use a tone that is informative yet engaging, and emphasize how this product enhances the user’s beauty routine. Keep the description concise and tailored for an audience seeking high-quality skincare or makeup solutions.",
            "Market Content": "Create a captivating marketing message for this L'Oreal cosmetic product that showcases the brand's innovation and commitment to beauty. The message should highlight the product's unique features, its benefits for the consumer, and how it reflects L'Oreal's values of luxury, science-backed beauty, and inclusivity. Use persuasive language that appeals to a broad audience, encouraging them to feel confident and empowered by using the product. Include a call-to-action that inspires engagement and aligns with L'Oreal's brand identity.",
            "Blog Creation": "Write an informative and engaging blog post about this L'Oreal cosmetic product. Begin with a brief introduction to the L'Oreal brand, emphasizing its leadership in beauty and innovation. Then, provide a detailed review of the product, highlighting its key features, benefits, and how it fits into a modern beauty routine. Include insights on the ingredients, results users can expect, and any tips on how to best use the product. Conclude with a call-to-action that encourages readers to try the product and experience the quality that L'Oreal is known for. The tone should be friendly and relatable while maintaining a sense of expertise in beauty care.",
            "Image caption": "Please provide a brief caption for this image.",
            "Detailed description": "Please provide a thoroughly detailed description of this image.",
            "Image classification": "Please categorize this image into one of the following categories: People, Food, Other. Only return the category name.",
            "Object recognition": "Please create a comma-separated list of the items found in this image. Only return the list of items.",
            "Writing a story": "Please write a fictional short story based on this image.",
            "Transcribing text": "Please transcribe any text found in this image. Then, translate it to spanish",
            "Translating text": "Please translate the text in this image to French.",
            "Other": "",
        }

        image_options_dict = {
            "Cosmetic": "images/aspect_resized_revitalift.jpg",
            "Food": "images/food.jpg",
            "People": "images/people.jpg",
            "Person and cat": "images/person_and_cat.jpg",
            "Room": "images/room.jpg",
            "Text in image": "images/text2.png",
            "Toy": "images/toy_car.jpg",
            "Other": "images/house.jpg",
        }

        with col1:
            st.subheader("Select an Image")
            image_selection = st.radio("Image example:", list(image_options_dict.keys()), key="image_selection_radio")
            uploaded_file = st.file_uploader("Select an image", type=['png', 'jpg'], key="ui_file_uploader_image_understanding") if image_selection == 'Other' else None

            if uploaded_file and image_selection == 'Other':
                uploaded_image_preview = ui_glib.get_bytesio_from_bytes(uploaded_file.getvalue())
                st.image(uploaded_image_preview)
            else:
                st.image(image_options_dict[image_selection])

    with col2:
        st.subheader("Prompt")
        
        # Add model selection
        model_options = {
            "Nova Lite": "amazon.nova-lite-v1:0",
            "Nova Pro": "amazon.nova-pro-v1:0",
            "Claude 3.5 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0"
        }
        
        selected_model = st.selectbox(
            "Select Model",
            list(model_options.keys())
        )
        
        prompt_selection = st.radio(
            "Prompt example:", 
            list(prompt_options_dict.keys()), 
            key="prompt_selection_radio"
        )
        
        prompt_text = st.text_area(
            "Prompt",
            value=prompt_options_dict[prompt_selection],
            height=100
        )

        go_button = st.button("Go", type="primary", key="go_button_image_understanding")

    with col3:
        st.subheader("Result")
        if go_button:
            with st.spinner("Processing..."):
                try:
                    image_bytes = uploaded_file.getvalue() if uploaded_file else ui_glib.get_bytes_from_file(image_options_dict[image_selection])
                    response = ui_glib.get_response_from_model(
                        prompt_content=prompt_text,
                        image_bytes=image_bytes,
                        model_id=model_options[selected_model]
                    )
                    st.write(response)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")









    # Subtab: Background Change
    with subtab3:
        st.title("Background Change(Outpainting)")

        col1, col2, col3 = st.columns(3)

        with col1:
            uploaded_file = st.file_uploader("Select an image", type=['png', 'jpg'], key="bg_file_uploader_background_change")
            if uploaded_file:
                uploaded_image_preview = bg_glib.get_bytesio_from_bytes(uploaded_file.getvalue())
                st.image(uploaded_image_preview)
            else:
                st.image("images/example.jpg")

        with col2:
            st.subheader("Image parameters")
            model_options = ["Nova Canvas", "Titan Image Generator G1 (v2)", "Titan Image Generator G1 (v1)"]
            selected_model = st.selectbox("Select Model", model_options, key="bg_model_selection")
            mask_prompt = st.text_input("Object to keep:", value="Clear L'Oreal bottle with white circular top")

            model_id_mapping = {
                "Nova Canvas": "amazon.nova-canvas-v1:0",
                "Titan Image Generator G1 (v2)": "amazon.titan-image-generator-v2:0",
                "Titan Image Generator G1 (v1)": "amazon.titan-image-generator-v1"
            }
            prompt_text = st.text_area("Description including the object to keep and background to add:", value="Cosmetic bottle on a marble shelf, with bathroom items in the back blurred out", height=100)
            negative_prompt = st.text_input("What should not be in the background:", value="low resolution")
            outpainting_mode = st.radio("Outpainting mode:", ["DEFAULT", "PRECISE"], horizontal=True, key="outpainting_mode_radio")
            generate_button = st.button("Generate", type="primary", key="generate_button_background_change")

        with col3:
            st.subheader("Result")
            if generate_button:
                with st.spinner("Drawing..."):
                    try:
                        image_bytes = uploaded_file.getvalue() if uploaded_file else bg_glib.get_bytes_from_file("images/example.jpg")
                        selected_model_id = model_id_mapping[selected_model]
                        generated_image = bg_glib.get_image_from_model(
                            prompt_content=prompt_text, 
                            image_bytes=image_bytes,
                            mask_prompt=mask_prompt,
                            negative_prompt=negative_prompt,
                            outpainting_mode=outpainting_mode,
                            model_id=selected_model_id
                        )
                        if generated_image:
                            st.image(generated_image)
                        else:
                            st.error("Failed to generate image. Please try again.")
                    except Exception as e:
                        st.error(f"Error during image generation: {str(e)}")






    # Subtab: Image Extension
    with subtab4:
        st.title("Image Extension(Outpainting)")

        # Add model selection at the top
        model_options = {
            "Amazon Titan V2": "amazon.titan-image-generator-v2:0",
            "Amazon Titan V1": "amazon.titan-image-generator-v1",
            "Nova Canvas": "amazon.nova-canvas-v1:0"
        }
        selected_model = st.selectbox("Select Model", list(model_options.keys()))
        model_id = model_options[selected_model]

        col1, col2, col3 = st.columns(3)

        horizontal_alignment_dict = {
            "Left": 0.0,
            "Center": 0.5,
            "Right": 1.0,
        }

        vertical_alignment_dict = {
            "Top": 0.0,
            "Middle": 0.5,
            "Bottom": 1.0,
        }

        horizontal_alignment_options = list(horizontal_alignment_dict)
        vertical_alignment_options = list(vertical_alignment_dict)

        with col1:
            st.subheader("Initial image")

            uploaded_file = st.file_uploader("Select an image (smaller than 1024x1024)", type=['png', 'jpg'])

            if uploaded_file:
                uploaded_image_preview = img_extension_glib.get_bytesio_from_bytes(uploaded_file.getvalue())
                st.image(uploaded_image_preview)
            else:
                st.image("images/example.jpg")

        with col2:
            st.subheader("Extension parameters")
            prompt_text = st.text_area("What should be seen in the extended image:", height=100, value="Flower pot", help="The prompt text")
            negative_prompt = st.text_input("What should not be in the extended area:", help="The negative prompt")

            horizontal_alignment_selection = st.select_slider("Original image horizontal placement:", options=horizontal_alignment_options, value="Center")
            vertical_alignment_selection = st.select_slider("Original image vertical placement:", options=vertical_alignment_options, value="Middle")

            # Add outpainting mode selection
            outpainting_mode = st.selectbox(
                "Outpainting Mode",
                ["DEFAULT", "PRECISE"],
                help="DEFAULT softens the mask. PRECISE keeps it sharp."
            )

            # Add quality selection
            quality_option = st.selectbox(
                "Quality",
                ["premium", "standard"],
                help="Select the quality of the generated image"
            )

            generate_button = st.button("Generate", type="primary", key="generate_button_image_extension")

        with col3:
            st.subheader("Result")

            if generate_button:
                with st.spinner("Drawing..."):
                    if uploaded_file:
                        image_bytes = uploaded_file.getvalue()
                    else:
                        image_bytes = img_extension_glib.get_bytes_from_file("images/example.jpg")

                    try:
                        # Assuming this returns bytes or a BytesIO object
                        generated_image = img_extension_glib.get_image_from_model(
                            prompt_content=prompt_text,
                            image_bytes=image_bytes,
                            negative_prompt=negative_prompt,
                            vertical_alignment=vertical_alignment_dict[vertical_alignment_selection],
                            horizontal_alignment=horizontal_alignment_dict[horizontal_alignment_selection],
                            model_id=model_id,
                            outpainting_mode=outpainting_mode,
                            quality=quality_option
                        )

                        # Convert the generated image to a PIL Image
                        if isinstance(generated_image, BytesIO):
                            img = Image.open(generated_image)  # Generated image is a BytesIO object
                        else:
                            img = Image.open(BytesIO(generated_image))  # If it's bytes, wrap in BytesIO for PIL

                        st.image(img)

                        # Prepare a buffer for the download button
                        buffered = BytesIO()
                        img.save(buffered, format="JPEG")
                        buffered.seek(0)

                        # Add a download button for the image
                        st.download_button(
                            label="Download Extended Image",
                            data=buffered,
                            file_name="extended_image.jpg",
                            mime="image/jpeg"
                        )

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with subtab5:
        st.title("Replace/Remove Image")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Image parameters")
            
            uploaded_file = st.file_uploader("Select an image", type=['png', 'jpg'])
            mask_prompt = ""
            
            if uploaded_file:
                uploaded_image_preview = rplce_rmv_glib.get_bytesio_from_bytes(uploaded_file.getvalue())
                st.image(uploaded_image_preview, caption="Uploaded Image")
                
                with st.spinner("Processing..."):
                    image_bytes = uploaded_file.getvalue()
                    identified_items = ui_glib.get_response_from_model(
                        prompt_content="Describe 1 object on the image. Be as detailed as possible, but stay within 3 to 5 words for the description.",
                        image_bytes=image_bytes
                    )
                    
                    if isinstance(identified_items, str):
                        mask_prompt = " ".join(identified_items.split())
                    elif isinstance(identified_items, list):
                        mask_prompt = " ".join(identified_items)
            else:
                st.image("images/aspect_resized_revitalift.jpg", caption="Example Image")
                st.warning("Please upload an image to generate a result.")
        
        with col2:
            # Select box for the model
            selected_model = st.selectbox("Select model", model_options, key="image_gen_model_selection")
            mask_prompt = st.text_input("Object to replace", value=mask_prompt.strip(), help="The mask text")
            prompt_text = st.text_area("Object to add (leave blank to remove)", value="", height=100, help="The prompt text")
            negative_prompt = st.text_input("What should not be in the result:", help="Optional negative prompt")
            generate_button = st.button("Generate", type="primary", key="generate_button_image_removal")
        
        with col3:
            st.subheader("Result")
            
            if generate_button:
                if not uploaded_file:
                    st.error("Please upload an image to generate the result.")
                else:
                    with st.spinner("Drawing..."):
                        image_bytes = uploaded_file.getvalue()
                        generated_image = rplce_rmv_glib.get_image_from_model(
                            prompt_content=prompt_text,
                            image_bytes=image_bytes,
                            mask_prompt=mask_prompt.strip(),
                            negative_prompt=negative_prompt,
                            model_id=model_options[selected_model]
                        )
                        
                        if generated_image:
                            st.image(generated_image, caption="Generated Image")
                        else:
                            st.error("Failed to generate image. Please try again.")

    with subtab6:
        st.header("Image Variation")
        
        # Model selection
        model_options = {
            "Nova Canvas": "amazon.nova-canvas-v1:0",
            "Amazon Titan V2": "amazon.titan-image-generator-v2:0"
        }
        selected_model = st.selectbox("Select Model", list(model_options.keys()), key="image_variation_model_select")
        model_id = model_options[selected_model]
        
        # Image upload
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
        
        # Display uploaded image
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(uploaded_file, use_column_width=True)
            with col2:
                st.subheader("Generated Variation")
                st.write("Variation will appear here after generation")
        
        # Text inputs
        prompt_text = st.text_input("Enter prompt for variation", 
                                placeholder="Example: Modernize the house, photo-realistic, 8k, hdr")
        negative_prompt = st.text_input("Enter negative prompt (optional)", 
                                    placeholder="Example: bad quality, low resolution, cartoon", value="None")
        
        # Optional parameters with sliders
        similarity_strength = st.slider("Similarity Strength", 0.2, 1.0, 0.7, 0.1)
        cfg_scale = st.slider("CFG Scale", 1.0, 20.0, 8.0, 0.5)
        
        # Image dimensions
        dim_col1, dim_col2 = st.columns(2)
        with dim_col1:
            width = st.number_input("Width", min_value=128, max_value=1024, value=512, step=128)
        with dim_col2:
            height = st.number_input("Height", min_value=128, max_value=1024, value=512, step=128)
        
        if st.button("Generate Variation") and uploaded_file is not None:
            with st.spinner("Generating image variation..."):
                try:
                    # Convert uploaded image to base64
                    image_bytes = uploaded_file.getvalue()
                    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Call backend function
                    result = img_variation_glib.generate_image_variation(
                        encoded_image,
                        prompt_text,
                        negative_prompt,
                        similarity_strength,
                        width,
                        height,
                        cfg_scale,
                        model_id
                    )
                    
                    # Display the result in the right column
                    col2.image(result, caption="Generated Variation", use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Error generating image: {str(e)}")








    # Subtab: Image Conditioning
    with subtab7:
        st.header("Image Conditioning")

        # Model selection
        model_options = {
            "Nova Canvas": "amazon.nova-canvas-v1:0",
            "Amazon Titan V1": "amazon.titan-image-generator-v1",
            "Amazon Titan V2": "amazon.titan-image-generator-v2:0"
        }
        selected_model = st.selectbox("Select Model", list(model_options.keys()))
        model_id = model_options[selected_model]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input")
            
            # Image upload with dynamic resolution message
            resolution_msg = "1280x720" if "nova" in model_id.lower() else "512x512"
            upload_msg = f"Upload an image (will be resized to {resolution_msg} if needed)"
            uploaded_file = st.file_uploader(upload_msg, type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file:
                try:
                    # Show original image and dimensions
                    original_image = Image.open(uploaded_file)
                    st.image(uploaded_file, caption=f"Original Image ({original_image.size[0]}x{original_image.size[1]})", use_column_width=True)
                    
                    # Check if resizing is needed based on model
                    target_size = (1280, 720) if "nova" in model_id.lower() else (512, 512)
                    if original_image.size != target_size:
                        st.info(f"Image will be automatically resized to {target_size[0]}x{target_size[1]} to meet model requirements.")
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

            # Text inputs
            prompt_text = st.text_area("Enter prompt:", height=100, 
                                    value="A robot playing soccer, anime cartoon style")
            negative_prompt = st.text_input("Enter negative prompt (optional):", 
                                        value="bad quality, low res")

            # Control parameters
            control_mode = st.selectbox("Control Mode", ["CANNY_EDGE", "SEGMENTATION"])
            control_strength = st.slider("Control Strength", 0.1, 1.0, 0.7, 0.1)

            generate_button = st.button("Generate", type="primary")

        with col2:
            st.subheader("Generated Image")
            
            if generate_button and uploaded_file:
                with st.spinner("Generating image..."):
                    try:
                        generated_image = img_condition_glib.generate_conditioned_image(
                            model_id=model_id,
                            image_bytes=uploaded_file.getvalue(),
                            prompt=prompt_text,
                            negative_prompt=negative_prompt,
                            control_mode=control_mode,
                            control_strength=control_strength
                        )
                        st.image(generated_image, caption="Generated Image", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error generating image: {str(e)}")






    # Subtab: Color Guided Content
    with subtab8:
        st.header("Color Guided Content")

        # Model selection
        model_options = {
            "Nova Canvas": "amazon.nova-canvas-v1:0",
            "Amazon Titan V2": "amazon.titan-image-generator-v2:0"
        }
        selected_model = st.selectbox("Select Model", list(model_options.keys()), key="color_guided_model_select")
        model_id = model_options[selected_model]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Parameters")
            
            # Text prompts
            prompt_text = st.text_area(
                "Enter prompt:",
                max_chars=512,
                height=100,
                help="Describe what you want to generate"
            )
            
            negative_prompt = st.text_area(
                "Enter negative prompt (optional):",
                max_chars=512,
                help="Describe what you don't want in the image"
            )

            # Color palette
            st.subheader("Color Palette")
            num_colors = st.slider("Number of colors", 1, 10, 3)
            colors = []
            for i in range(num_colors):
                color = st.color_picker(f"Color {i+1}", f"#{hex(i*20)[2:].zfill(6)}")
                colors.append(color)

            # Reference image
            st.subheader("Reference Image (Optional)")
            reference_image = st.file_uploader(
                "Upload an image (max 1408px)",
                type=['png', 'jpg', 'jpeg']
            )

            if reference_image:
                st.image(reference_image, caption="Reference Image")

            # Generation parameters
            st.subheader("Generation Parameters")
            col_w, col_h = st.columns(2)
            with col_w:
                width = st.select_slider(
                    "Width",
                    options=[512, 768, 1024, 1408],
                    value=1024
                )
            with col_h:
                height = st.select_slider(
                    "Height",
                    options=[512, 768, 1024, 1408],
                    value=1024
                )

            cfg_scale = st.slider("CFG Scale", 1.0, 20.0, 8.0, 0.5, key="cfg_scale_slider")
            num_images = st.slider("Number of Images", 1, 4, 1, key="num_images_slider")

            generate_button = st.button("Generate", type="primary", key="color_guided_generate_btn")

        with col2:
            st.subheader("Generated Images")
            
            if generate_button:
                if not prompt_text:
                    st.error("Please enter a prompt")
                elif not colors:
                    st.error("Please select at least one color")
                else:
                    with st.spinner("Generating images..."):
                        try:
                            images = clr_guided_glib.generate_images(
                                model_id=model_id,
                                prompt=prompt_text,
                                colors=colors,
                                negative_prompt=negative_prompt,
                                reference_image=reference_image,
                                width=width,
                                height=height,
                                num_images=num_images,
                                cfg_scale=cfg_scale
                            )
                            
                            for idx, img in enumerate(images):
                                st.image(img, caption=f"Generated Image {idx+1}")
                                
                                # Add download button for each image
                                buffered = BytesIO()
                                img.save(buffered, format="PNG")
                                st.download_button(
                                    label=f"Download Image {idx+1}",
                                    data=buffered.getvalue(),
                                    file_name=f"color_guided_image_{idx+1}.png",
                                    mime="image/png"
                                )
                        except Exception as e:
                            st.error(f"Error generating images: {str(e)}")







    # Subtab: Background Removal
    with subtab9:
        st.title("Background Removal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=['png', 'jpg', 'jpeg'],
                key="bg_removal_uploader"
            )
            
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image")
                
        with col2:
            st.subheader("Result")
            if uploaded_file:
                generate_button = st.button("Remove Background", type="primary")
                
                if generate_button:
                    with st.spinner("Processing..."):
                        try:
                            image_bytes = uploaded_file.getvalue()
                            generated_image = bg_removal_glib.remove_background(
                                image_bytes=image_bytes
                            )
                            if generated_image:
                                st.image(generated_image, caption="Result")
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")






    # Tab 2: Video Upload/Playback (Subtitle/Translation)
    with tab2:
        st.title("Video Upload and Playback")

        col1, col2 = st.columns(2)

        # Video upload to S3
        with col1:
            st.subheader("Upload and Preview Video")
            uploaded_video = st.file_uploader("Select a video", type=['mp4', 'mov', 'avi'], key="video_uploader")

            if uploaded_video:
                st.video(uploaded_video)
                video_filename = uploaded_video.name
                content_type = "video/mp4"
                # Compress video before upload
                compressed_video_data, compression_success = vid_analytics_lib.setup_compression()(uploaded_video.getvalue())
                if compression_success:
                    upload_status = sub_tran_glib.upload_file_to_s3(compressed_video_data, bucket_name, video_filename, content_type)
                else:
                    upload_status = sub_tran_glib.upload_file_to_s3(uploaded_video.getvalue(), bucket_name, video_filename, content_type)
                st.success(upload_status)
                st.session_state['video_filename'] = video_filename

    # Video rendering from S3
    with col2:
        st.subheader("Play and Process Video from S3")
        video_filename_input = st.text_input("S3 video filename to play:", value=st.session_state.get('video_filename', ""))
        language_mapping = {"Spanish": "es", "French": "fr"}
        selected_language = st.selectbox("Select Language", list(language_mapping.keys()))
        video_processed = st.session_state.get("video_processed", False)
        process_video_button = st.button("Process Video with Subtitles")
        play_button = st.button("Play Video with Subtitles", key="play_button")

        if process_video_button and video_filename_input:
            with st.spinner("Processing video..."):
                target_language_code = language_mapping[selected_language]
                result = sub_tran_glib.process_video_with_subtitles(
                    region="us-east-1",
                    inbucket=bucket_name,
                    infile=video_filename_input,
                    outbucket=bucket_name,
                    outfilename=video_filename_input.split('.')[0],
                    outfiletype="mp4",
                    target_language=target_language_code
                )
                st.success(result)
                st.session_state["video_processed"] = True

        if play_button and video_filename_input:
            subtitle_video_filename = f"{video_filename_input.split('.')[0]}_subtitle.mp4"
            video_url = sub_tran_glib.get_video_url_from_s3(bucket_name, subtitle_video_filename)
            st.video(video_url)
            st.success(f"Playing video with subtitles from S3: {subtitle_video_filename}")





    # Tab 3: Image/Text to Video Generation
    with tab3:
        st.title("Video Generation - LLM: Amazon Nova Reel")

        generation_type = st.radio(
            "Select Generation Type",
            ["Text to Video", "Image to Video"],
            horizontal=True,
            key="generation_type_radio"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input")

            # Text prompt input
            prompt_text = st.text_area(
                "Enter your video description:",
                value="A sun rising on a beachfront in hawaii, with palm trees moving  in the background.",
                height=100
            )

            # Image upload and camera capture (only shown for image-to-video)
            uploaded_image = None
            if generation_type == "Image to Video":
                # Create container for image input options
                image_input_container = st.container()
                
                with image_input_container:
                    image_source = st.radio(
                        "Choose image source",
                        ["Upload Image", "Take Picture"],
                        horizontal=True,
                        key="image_source_radio"
                    )

                    if image_source == "Upload Image":
                        uploaded_image = st.file_uploader(
                            "Upload starting image",
                            type=['png', 'jpg', 'jpeg'],
                            key="video_gen_image_upload"
                        )
                    else:  # Take Picture option
                        camera_image = st.camera_input(
                            "Take a picture",
                            key="video_gen_camera"
                        )
                        if camera_image:
                            uploaded_image = camera_image

                    if uploaded_image:
                        st.image(uploaded_image, caption="Starting Image", use_column_width=True)

            # Video generation parameters
            st.subheader("Parameters")
            duration = 6
            fps = st.selectbox("Frames Per Second", [24], index=0)
            dimension = st.selectbox(
                "Resolution",
                ["1280x720"],
                index=0
            )

            generate_button = st.button("Generate Video", type="primary")

        with col2:
            st.subheader("Generated Video")

            if generate_button:
                with st.spinner("Generating video..."):
                    try:
                        # Create progress indicator
                        progress_text = "Operation in progress. Please wait..."
                        my_bar = st.progress(0, text=progress_text)

                        model_input = {
                            "taskType": "TEXT_VIDEO",
                            "textToVideoParams": {
                                "text": prompt_text
                            },
                            "videoGenerationConfig": {
                                "durationSeconds": duration,
                                "fps": fps,
                                "dimension": dimension,
                                "seed": 0
                            }
                        }

                        # Add image if provided
                        if uploaded_image:
                            # Get image format
                            image_format = uploaded_image.name.split('.')[-1].lower()
                            
                            # Resize image if needed
                            resize_result = img_to_video_glib.resize_image_to_1280x720(uploaded_image.getvalue())
                            
                            if resize_result['status'] == 'error':
                                st.error(f"Error processing image: {resize_result['message']}")
                            else:
                                # Show resize notification if image was modified
                                if resize_result['was_resized']:
                                    st.info(f"Image was automatically resized from {resize_result['original_size']} to 1280x720 to meet model requirements.")
                                
                                base64_image = base64.b64encode(resize_result['image_bytes']).decode('utf-8')
                                model_input["textToVideoParams"]["images"] = [{
                                    "format": image_format if image_format != 'jpg' else 'jpeg',
                                    "source": {
                                        "bytes": base64_image
                                    }
                                }]

                        # Start async video generation
                        response = bedrock_client.start_async_invoke(
                            modelId="amazon.nova-reel-v1:0",
                            modelInput=model_input,
                            outputDataConfig={
                                "s3OutputDataConfig": {
                                    "s3Uri": f"s3://{bucket_name}"
                                }
                            }
                        )

                        # Get the invocation ARN from the response
                        invocation_arn = response['invocationArn']
                        if not invocation_arn:
                            raise ValueError("No invocation ARN in response")

                        st.success(f"Video generation started! ARN: {invocation_arn}")

                        # Polling for job completion
                        status = "InProgress"
                        poll_count = 0
                        while status == "InProgress" and poll_count < 20:
                            time.sleep(15)
                            status_response = bedrock_client.get_async_invoke(
                                invocationArn=invocation_arn
                            )
                            status = status_response['status']
                            poll_count += 1
                            my_bar.progress(poll_count * 5, f"Status: {status}")

                        if status == "Completed":
                            my_bar.progress(100, "Completed!")
                            st.success("Video generation completed!")
                            
                            # Get the S3 location from the invocation ARN
                            s3_prefix = invocation_arn.split('/')[-1]
                            object_key = f"{s3_prefix}/output.mp4"
                            
                            # Generate presigned URL for the video
                            presigned_url = txt_to_video_glib.get_presigned_url(bucket_name, object_key)
                            
                            if presigned_url:
                                st.video(presigned_url)
                            else:
                                st.error("Failed to generate video URL.")

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                    finally:
                        pass






    # Tab 4: Video Analytics
    with tab4:
        st.title("Video Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload and Preview Video")
            uploaded_video = st.file_uploader(
                "Select a video for analysis",
                type=['mp4', 'mov', 'avi'],
                key="analytics_video_uploader"
            )
            
            if uploaded_video:
                st.video(uploaded_video)
                video_filename = uploaded_video.name
                content_type = "video/mp4"
                upload_status = vid_analytics_lib.upload_file_to_s3(
                    uploaded_video.getvalue(),
                    bucket_name,
                    video_filename,
                    content_type
                )
                st.success(upload_status)
                st.session_state['analytics_video_filename'] = video_filename


        with col2:
            st.subheader("Video Analysis Results")
            
            # Model selection
            model_options = {
                "Amazon Nova Lite": "us.amazon.nova-lite-v1:0",
                "Amazon Nova Pro": "amazon.nova-pro-v1:0"
            }
            selected_model = st.selectbox(
                "Select Model",
                list(model_options.keys())
            )
            
            # Add voice selection for audio
            voice_options = {
                "Matthew (Male)": "Matthew",
                "Joanna (Female)": "Joanna",
                "Stephen (Male)": "Stephen",
                "Kendra (Female)": "Kendra"
            }
            selected_voice = st.selectbox(
                "Select Voice for Audio",
                list(voice_options.keys())
            )
            
            video_filename_input = st.text_input(
                "S3 video filename to analyze:",
                value=st.session_state.get('analytics_video_filename', "")
            )
            
            analysis_prompt = st.text_area(
                "Enter analysis prompt:",
                value="Analyze this video and provide a detailed description of its content, including any key scenes, actions, or notable elements.",
                height=100
            )
            
            analyze_button = st.button("Analyze Video", key="analyze_video_button")
            
            if analyze_button and video_filename_input:
                with st.spinner(f"Analyzing video with {selected_model}..."):
                    try:
                        progress_text = "Analysis in progress. Please wait..."
                        progress_bar = st.progress(0, text=progress_text)
                        
                        analysis_result = vid_analytics_lib.analyze_video_with_nova(
                            bucket_name,
                            video_filename_input,
                            analysis_prompt,
                            model_options[selected_model]
                        )
                        
                        if analysis_result['status'] == 'success':
                            progress_bar.progress(100, "Analysis completed!")
                            st.success("Video analysis completed successfully!")
                                                            
                            # Store analysis in session state
                            st.session_state['current_analysis'] = analysis_result['analysis']
                            
                            analysis_text = analysis_result['analysis']

                            st.download_button(
                                label="Download Analysis",
                                data=analysis_text,
                                file_name="video_analysis.txt",
                                mime="text/plain"
                            )
                        else:
                            st.error(f"Analysis failed: {analysis_result['message']}")
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

            # Show analysis from session state if it exists
            if 'current_analysis' in st.session_state:
                st.subheader("Video Analysis")
                st.write(st.session_state['current_analysis'])
                
                # Handle audio generation separately
                listen_button = st.button("Generate Audio", key="listen_analysis_button")
                if listen_button:
                    with st.spinner("Generating audio with Amazon Polly..."):
                        try:
                            audio_result = vid_analytics_lib.generate_audio(
                                bucket_name,
                                st.session_state['current_analysis'],
                                voice_options[selected_voice]
                            )
                            if audio_result['status'] == 'success':
                                st.audio(audio_result['audio_url'])
                            else:
                                st.error(f"Audio generation failed: {audio_result['message']}")
                        except Exception as e:
                            st.error(f"Error during audio generation: {str(e)}")
