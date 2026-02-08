import streamlit as st
from huggingface_hub import InferenceClient
import tempfile
import os

st.title("Veloce 2.0")

api_token = st.text_input("Your Hugging Face API Token", type="password")

prompt = st.text_area("Enter your video prompt (e.g., 'A serene mountain lake at sunrise with birds flying and gentle wind sounds')", height=120)

# Aspect ratio selector
aspect_ratio = st.radio("Aspect Ratio", ["16:9 (Landscape)", "9:16 (Portrait)"], index=0)

# Duration selector
duration_sec = st.radio("Duration", ["10 seconds", "15 seconds"], index=0)

if st.button("Generate Video"):
    if not api_token.strip():
        st.error("Please enter your Hugging Face API token.")
    elif not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Generating video with synced audio... (1-10+ min depending on load)"):
            try:
                client = InferenceClient(token=api_token)

                # Map selections to params
                if aspect_ratio == "16:9 (Landscape)":
                    width, height = 768, 432  # Balanced for many providers; increase to 1280x720 if it works
                else:  # 9:16
                    width, height = 432, 768

                if duration_sec == "10 seconds":
                    num_frames = 250  # Approx 25 fps
                else:
                    num_frames = 375  # Approx 25 fps for 15s

                # Generate using LTX-2 (outputs MP4 with native audio)
                video_bytes = client.text_to_video(
                    prompt=prompt,
                    model="Lightricks/LTX-2",
                    num_inference_steps=25,     # 20-40 range for balance
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    guidance_scale=7.0
                )

                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(video_bytes)
                    video_path = tmp_file.name

                # Display the result
                st.video(video_path)
                st.success("Video generated successfully! (includes synced audio)")

                # Optional: clean up after display (comment out if you want to keep files)
                # os.remove(video_path)

            except Exception as e:
                st.error(f"Generation error: {str(e)}\n\n"
                         "Tips: \n"
                         "- Double-check your token is valid and has Inference API access.\n"
                         "- Try a shorter/simpler prompt or the 10-second option first.\n"
                         "- Some providers queue large models—wait and retry.\n"
                         "- Test the model directly in a Hugging Face Space (search 'LTX-2') to confirm.")
