"""
Storytelling Application for Kids
--------------------------------
- Upload an image -> automatic caption -> generate a 50-100 word children's story -> text-to-speech audio.
- Uses Hugging Face models and gTTS.
- Designed for Streamlit Cloud deployment; secrets can store HF_TOKEN if needed.
"""

import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import io


# -------------------------------------------------------------------
# 1. Model loading (cached to avoid reloading on every interaction)
# -------------------------------------------------------------------
@st.cache_resource
def load_caption_model():
    """Load BLIP image-to-text model."""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@st.cache_resource
def load_story_model():
    """
    Load FLAN-T5-small for instruction-based story generation.
    If a Hugging Face token is stored in secrets, it can be passed
    (though this model is public and does not require a token).
    """
    token = None
    if "HF_TOKEN" in st.secrets:
        token = st.secrets["HF_TOKEN"]
    # device=-1 forces CPU; Streamlit Cloud free tier has no GPU.
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1,
        use_auth_token=token
    )


# -------------------------------------------------------------------
# 2. Core functions (as required by the skeleton)
# -------------------------------------------------------------------
def img2text(image):
    """
    Convert a PIL image to a text caption.
    Args:
        image (PIL.Image): Input image.
    Returns:
        str: Generated caption.
    """
    caption_pipeline = load_caption_model()
    result = caption_pipeline(image)
    caption = result[0]["generated_text"]
    return caption


def text2story(caption):
    """
    Expand a short caption into a full children's story (50-100 words).
    Args:
        caption (str): The image caption.
    Returns:
        str: Generated story.
    """
    story_pipeline = load_story_model()
    prompt = (
        f"Write a short children's story of 50 to 100 words based on this description: {caption}"
    )
    # max_length includes prompt + new tokens. Prompt ~20 tokens, so ~130 tokens for story (~100 words)
    output = story_pipeline(prompt, max_length=150, do_sample=False)
    story = output[0]["generated_text"]
    return story.strip()


def text2audio(story_text):
    """
    Convert story text to audio (MP3) using Google Text-to-Speech.
    Args:
        story_text (str): The story to read.
    Returns:
        io.BytesIO: In-memory bytes of the MP3 audio.
    """
    tts = gTTS(text=story_text, lang="en", slow=False)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes


# -------------------------------------------------------------------
# 3. Streamlit UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Storytelling App for Kids", page_icon="📖")
    st.title("✨ Storytelling App ✨")
    st.markdown(
        "Upload an image, and I will create a magical 50–100 word children's story and read it aloud!"
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Story"):
            # Step 1: Caption the image
            with st.spinner("Looking at the image..."):
                caption = img2text(image)
                st.info(f"📷 *What I see:* {caption}")

            # Step 2: Generate story
            with st.spinner("Writing a story just for you..."):
                story = text2story(caption)
                word_count = len(story.split())
                st.success(f"📖 *Your Story* ({word_count} words):")
                st.write(story)

                # Optional: notify if length deviates (model is usually on target)
                if word_count < 50:
                    st.warning("The story is a bit short. Let's imagine the rest!")
                elif word_count > 100:
                    st.info("The story is a bit long. Enjoy the extra magic!")

            # Step 3: Convert to audio
            with st.spinner("Converting story to audio..."):
                audio_bytes = text2audio(story)
                st.audio(audio_bytes, format="audio/mp3")
                st.success("🎧 Listen to your story above!")


if __name__ == "__main__":
    main()
