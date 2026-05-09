import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
from gtts import gTTS
import io
import torch
import time
import re


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
# 2. Story Generation with FLAN‑T5‑Small (Optimized for Speed & Coherence)
# -------------------------------------------------------------------
@st.cache_resource
def load_flan_small():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

def remove_repetition(text):
    """Remove repeated phrases algorithmically (no templates)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    unique_sentences = []
    seen = set()
    for sent in sentences:
        sent_lower = sent.lower().strip()
        if sent_lower and sent_lower not in seen:
            seen.add(sent_lower)
            unique_sentences.append(sent)
    result = ' '.join(unique_sentences)
    # If result is too short, return original (but normally it's fine)
    if len(result.split()) < 10 and len(text.split()) > 20:
        return text
    return result

def text2story(caption):
    """
    Generate a story using FLAN-T5-small with a structured prompt.
    No templates – the model generates every word based on the prompt.
    """
    tokenizer, model = load_flan_small()
    
    # The trick: force a story structure by starting the continuation
    prompt = (
        f"Write a children's story about {caption}. "
        f"Story: Once upon a time,"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            min_new_tokens=50,
            do_sample=True,
            temperature=0.85,
            repetition_penalty=1.3,       # Strong penalty to avoid loops
            no_repeat_ngram_size=3,
            num_beams=4,
            early_stopping=True
        )
    
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt part (if it appears)
    if story.startswith(prompt):
        story = story[len(prompt):]
    elif story.startswith("Once upon a time,"):
        story = story  # keep as is
    else:
        # Prepend "Once upon a time" if missing
        story = "Once upon a time, " + story
    
    # Clean up: ensure first letter capital, end with period
    story = story.strip()
    if story and story[0].islower():
        story = story[0].upper() + story[1:]
    if story and story[-1] not in '.!?':
        story += '.'
    
    # Remove any repetitive cycles (algorithmic)
    story = remove_repetition(story)
    
    # Trim to 50-100 words (only trim, no addition)
    words = story.split()
    if len(words) > 100:
        story = ' '.join(words[:100])
        if not story.endswith(('.', '!', '?')):
            story += '.'
    
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
        "Upload an image. AI describes it, then **generates an original children's story** (50–100 words). "
        "The story is created entirely by a text generation model – no pre‑written sentences."
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Story"):
            with st.spinner("📷 Describing the image..."):
                caption = img2text(image)
                st.info(f"📷 *What I see:* {caption}")

            with st.spinner("📝 Writing a story (5-10 seconds)..."):
                story = text2story(caption)
                word_count = len(story.split())
                st.success(f"📖 *Your Story* ({word_count} words):")
                st.write(story)

                if word_count < 50:
                    st.warning("The story is a bit short – the AI kept it concise.")
                elif word_count > 100:
                    st.info("A longer story – enjoy!")

            with st.spinner("🔊 Preparing audio..."):
                audio_bytes, fallback_html = text2audio(story)
                if audio_bytes is not None:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success("🎧 Listen to your story above!")
                else:
                    st.warning("⚠️ Auto TTS unavailable – using browser voice.")
                    st.components.v1.html(fallback_html, height=0)
                    st.markdown(
                        f"""
                        <button onclick="window.speechSynthesis.speak(new SpeechSynthesisUtterance(`{story.replace('`', '\\`')}`))">
                            🔊 Play Story
                        </button>
                        """,
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
