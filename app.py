import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
from gtts import gTTS
import io
import torch
import time


# -------------------------------------------------------------------
# 1. Image Captioning (BLIP from Hugging Face)
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
# 2. Story Generation with GPT-2 (distilgpt2 – small, fast, coherent)
# -------------------------------------------------------------------
@st.cache_resource
def load_story_generator():
    """Load a text generation pipeline with GPT-2 (distilled version for speed)."""
    return pipeline(
        "text-generation",
        model="distilgpt2",
        device=-1  # CPU
    )

def text2story(caption):
    """
    Generate a 50-100 word children's story from the image caption using GPT-2.
    """
    generator = load_story_generator()
    
    # Prompt engineering: prime the model to write a story
    prompt = f"Once upon a time, there was {caption}. "
    prompt += "This is a short children's story about what happened next: "
    
    # Generate with parameters that discourage repetition
    output = generator(
        prompt,
        max_new_tokens=150,
        min_new_tokens=60,
        do_sample=True,
        temperature=0.85,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=generator.tokenizer.eos_token_id,
        pad_token_id=generator.tokenizer.eos_token_id,
        truncation=True
    )
    
    # Extract generated text and remove the prompt part
    full_text = output[0]['generated_text']
    story = full_text[len(prompt):].strip()
    
    # If the story is too short or empty, fallback to a simple completion
    if len(story.split()) < 30:
        # Fallback: use a more deterministic generation
        story = simple_story_fallback(caption)
    
    # Ensure length between 50-100 words
    words = story.split()
    if len(words) > 100:
        story = ' '.join(words[:100]) + " The end."
    elif len(words) < 50:
        story = story + " They were all very happy. And they lived happily ever after. The end."
    
    return story

def simple_story_fallback(caption):
    """
    A very simple deterministic story generator when GPT-2 fails.
    Still uses the caption, but with a template to guarantee coherence.
    """
    templates = [
        f"{caption} One day, they decided to go on a picnic. They packed sandwiches and juice. "
        f"The sun was shining, and the birds were singing. They had a wonderful time and returned home "
        f"with happy hearts. The end.",
        
        f"{caption} Everyone in the neighborhood loved them. One morning, a little friend came to visit. "
        f"They played together all day long. They shared toys and laughed a lot. "
        f"It was the best day ever. They promised to be friends forever. The end."
    ]
    import random
    return random.choice(templates)


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
        "Upload an image. I will use AI to describe it and then write a magical children's story (50–100 words) and read it aloud!"
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Story"):
            with st.spinner("Looking at the image..."):
                caption = img2text(image)
                st.info(f"📷 *What I see:* {caption}")

            with st.spinner("Writing a story using AI text generation..."):
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
                    st.warning("Auto TTS temporarily unavailable. Using browser speech.")
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
