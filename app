"""
Storytelling App for Children (3–10 years)
-------------------------------------------
1. User uploads an image.
2. A caption is generated using a pre‑trained image‑to‑text model.
3. The caption is expanded into a short, fun story (50–100 words).
4. The story is converted to speech and played automatically.
5. Built with Streamlit and Hugging Face Transformers.
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import io

# ------------------------------
# Cache model loading for performance
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_captioning_model():
    """Load the image captioning pipeline from Hugging Face."""
    try:
        # Using a relatively lightweight model (vit‑gpt2) to stay within memory limits
        captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        return captioner
    except Exception as e:
        st.error(f"Failed to load captioning model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_text_gen_model():
    """Load the text generation pipeline from Hugging Face."""
    try:
        # distilgpt2 is small and fast, perfect for short children's stories
        generator = pipeline("text-generation", model="distilgpt2")
        return generator
    except Exception as e:
        st.error(f"Failed to load text generation model: {e}")
        return None

# ------------------------------
# Helper functions
# ------------------------------
def generate_caption(captioner, image):
    """
    Generate a textual description (caption) of the uploaded image.
    Returns a string.
    """
    try:
        result = captioner(image)
        caption = result[0]['generated_text']
        return caption
    except Exception as e:
        st.error(f"Caption generation failed: {e}")
        return "a beautiful scene"

def generate_story(generator, caption):
    """
    Expand the caption into a full children's story of 50–100 words.
    Uses a simple prompt to guide the model towards appropriate content.
    """
    # Prompt designed for young children – fun, simple, and positive
    prompt = (
        f"Once upon a time, {caption}. "
        f"Write a very short, happy, and simple children's story (about 50–100 words):"
    )
    try:
        output = generator(
            prompt,
            max_new_tokens=150,          # ~ 100 words
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=50256,          # distilgpt2's eos token as pad
            return_full_text=False       # only return the newly generated text
        )
        story = output[0]['generated_text'].strip()

        # Clean up common artefacts
        story = story.replace("  ", " ")

        # Ensure minimum length (if too short, add a friendly note)
        if len(story.split()) < 30:
            story += " They all lived happily ever after."

        # Trim to about 100 words maximum (for clear audio)
        words = story.split()
        if len(words) > 120:
            story = " ".join(words[:120]) + "."

        return story
    except Exception as e:
        st.error(f"Story generation failed: {e}")
        return f"{caption} And everyone had a wonderful time. The end."

def text_to_speech(text):
    """
    Convert the story text into an MP3 audio stream using gTTS.
    Returns a BytesIO object containing the audio data.
    """
    try:
        tts = gTTS(text, lang="en", slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Text‑to‑speech conversion failed: {e}")
        return None

# ------------------------------
# Streamlit UI
# ------------------------------
def main():
    st.set_page_config(page_title="Kids Storyteller", page_icon="📖")
    st.title("🖼️📖 Storytelling App for Kids")
    st.markdown(
        "**Upload a picture** and I will turn it into a short, fun story with audio! "
        "(Best for children 3–10 years old)"
    )

    # Sidebar instructions
    with st.sidebar:
        st.header("How to use")
        st.write("1. Upload a clear image (JPG/PNG).")
        st.write("2. Click **Generate Story & Audio**.")
        st.write("3. Read the story and listen to the audio.")
        st.info("Models are downloaded once – first run may take 1–2 minutes.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Your uploaded picture", use_container_width=True)

        # Main action button
        if st.button("✨ Generate Story & Audio", type="primary"):
            with st.spinner("Loading smart models (first time might be slower)..."):
                captioner = load_captioning_model()
                generator = load_text_gen_model()

            if captioner is None or generator is None:
                st.stop()

            # Step 1: Caption
            with st.spinner("📝 Describing the image..."):
                caption = generate_caption(captioner, image)

            # Step 2: Story generation
            with st.spinner("✍️ Writing a short story for you..."):
                story = generate_story(generator, caption)

            # Step 3: Display story
            st.subheader("✨ Your Story")
            st.write(story)
            st.caption(f"📊 Word count: {len(story.split())}")

            # Step 4: Text‑to‑Speech
            with st.spinner("🔊 Turning story into speech..."):
                audio_data = text_to_speech(story)

            if audio_data:
                st.subheader("🔊 Listen to the story")
                st.audio(audio_data, format="audio/mp3")
                # Optional: provide a download button for the audio
                st.download_button(
                    label="⬇️ Download audio (MP3)",
                    data=audio_data,
                    file_name="story_audio.mp3",
                    mime="audio/mpeg"
                )

            st.success("Enjoy your story! 🎉")
    else:
        st.info("👆 Please upload an image to begin.")

if __name__ == "__main__":
    main()
