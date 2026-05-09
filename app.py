import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
from gtts import gTTS
import io
import torch
import time
import random


# -------------------------------------------------------------------
# 1. Image Captioning (BLIP)
# -------------------------------------------------------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def img2text(image):
    processor, model = load_blip()
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# -------------------------------------------------------------------
# 2. Story Generation (FLAN-T5 with fallback template)
# -------------------------------------------------------------------
@st.cache_resource
def load_flan():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

def generate_story_with_flan(caption):
    """Use FLAN-T5 to generate a story. Returns story string or None if failed."""
    try:
        tokenizer, model = load_flan()
        prompt = (
            f"Write a very short children's story of 50-70 words based on: '{caption}'. "
            f"The story must have a clear beginning, middle, and end. "
            f"Use simple words. Example: 'Once upon a time, ... The end.'"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                min_length=60,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                num_beams=4,
                repetition_penalty=1.2
            )
        story = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up common issues
        story = story.replace("  ", " ").strip()
        if len(story.split()) < 30:
            return None  # fallback
        return story
    except Exception:
        return None

def fallback_story(caption):
    """Create a simple, coherent story from the caption using templates."""
    templates = [
        f"Once upon a time, there was a {caption}. It was a bright and sunny day. "
        f"The {caption} was very happy and wanted to explore. It met new friends along the way. "
        f"They played and laughed together. At the end of the day, the {caption} felt grateful. "
        f"And they all lived happily ever after. The end.",
        
        f"One day, a little child saw a {caption}. The child was amazed by its beauty. "
        f"The {caption} smiled and said, 'Let's be friends!' They went on an adventure in the forest. "
        f"They found a hidden treasure of sparkling gems. They shared the treasure with everyone. "
        f"What a wonderful day it was! The end.",
        
        f"In a magical land, there lived a {caption}. Every morning, the {caption} would wake up and sing. "
        f"The birds and butterflies loved the song. One day, a tiny fairy visited the {caption}. "
        f"The fairy granted a wish: to make everyone smile. The {caption} wished for kindness. "
        f"From that day on, the land was full of love and joy. The end."
    ]
    return random.choice(templates)

def text2story(caption):
    """Main story function: try FLAN, fallback to template."""
    story = generate_story_with_flan(caption)
    if story is None:
        story = fallback_story(caption)
    # Ensure length between 50-100 words
    words = story.split()
    if len(words) < 50:
        story += " " + fallback_story(caption).split()[-50:]  # append ending from another template
    elif len(words) > 100:
        story = " ".join(words[:100]) + " The end."
    return story


# -------------------------------------------------------------------
# 3. Text-to-Speech (gTTS with retry + browser fallback)
# -------------------------------------------------------------------
def text2audio_gtss(story_text):
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            tts = gTTS(text=story_text, lang="en", slow=False)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            return audio_bytes
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                raise
    return None

def text2audio_fallback(story_text):
    safe_text = story_text.replace("'", "\\'")
    js_code = f"""
        <script>
            (function() {{
                var utterance = new SpeechSynthesisUtterance('{safe_text}');
                utterance.lang = 'en-US';
                utterance.rate = 0.9;
                window.speechSynthesis.speak(utterance);
            }})();
        </script>
    """
    return js_code

def text2audio(story_text):
    try:
        audio_bytes = text2audio_gtss(story_text)
        return audio_bytes, None
    except Exception:
        fallback_html = text2audio_fallback(story_text)
        return None, fallback_html


# -------------------------------------------------------------------
# 4. Streamlit UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Storytelling App for Kids", page_icon="📖")
    st.title("✨ Storytelling App ✨")
    st.markdown(
        "Upload an image, and I will create a magical children's story (50–100 words) and read it aloud!"
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Story"):
            with st.spinner("Looking at the image..."):
                caption = img2text(image)
                st.info(f"📷 *What I see:* {caption}")

            with st.spinner("Writing a story..."):
                story = text2story(caption)
                word_count = len(story.split())
                st.success(f"📖 *Your Story* ({word_count} words):")
                st.write(story)

                if word_count < 50:
                    st.warning("The story is a bit short. Let's imagine the rest!")
                elif word_count > 100:
                    st.info("The story is a bit long. Enjoy the extra magic!")

            with st.spinner("Preparing audio..."):
                audio_bytes, fallback_html = text2audio(story)
                if audio_bytes is not None:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success("🎧 Listen to your story above!")
                else:
                    st.warning("Auto TTS unavailable – using browser speech.")
                    st.components.v1.html(fallback_html, height=0)
                    st.markdown(
                        f"""
                        <button onclick="window.speechSynthesis.speak(new SpeechSynthesisUtterance(`{story.replace('`', '\\`')}`))">
                            🔊 Play Story (Manual)
                        </button>
                        """,
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
