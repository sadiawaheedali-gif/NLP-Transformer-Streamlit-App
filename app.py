import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Multilingual NLP Application",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# HUGGING FACE REPOSITORIES
# =========================================================

MBERT_MODEL = "Soho77/mbert-xnli-results"
MBERT_TOKENIZER = "Soho77/mbert-xnli-multilingual"

GPT2_MODEL = "Soho77/gpt2-balochi-results"
GPT2_TOKENIZER = "Soho77/gpt2-balochi-generation"


# =========================================================
# DEVICE
# =========================================================

# Use GPU if available, otherwise CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# LOAD mBERT ONLY ONCE
# =========================================================

@st.cache_resource(show_spinner="Loading mBERT model...")

def load_mbert():

    tokenizer = AutoTokenizer.from_pretrained(
        MBERT_TOKENIZER
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MBERT_MODEL
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# =========================================================
# LOAD GPT-2 ONLY ONCE
# =========================================================

@st.cache_resource(show_spinner="Loading GPT-2 model...")

def load_gpt2():

    tokenizer = AutoTokenizer.from_pretrained(
        GPT2_TOKENIZER
    )

    model = AutoModelForCausalLM.from_pretrained(
        GPT2_MODEL
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# =========================================================
# mBERT PREDICTION
# =========================================================

def predict_nli(premise, hypothesis):

    tokenizer, model = load_mbert()

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[0, predicted_class].item()

    # XNLI labels
    labels = {
        0: "ENTAILMENT",
        1: "NEUTRAL",
        2: "CONTRADICTION"
    }

    label = labels.get(
        predicted_class,
        f"LABEL_{predicted_class}"
    )

    return label, confidence


# =========================================================
# GPT-2 GENERATION
# =========================================================

def generate_text(prompt):

    tokenizer, model = load_gpt2()

    formatted_prompt = (
        "Instruction: "
        + prompt
        + "\nResponse:"
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        output = model.generate(
            **inputs,

            max_new_tokens=60,

            do_sample=True,

            temperature=0.7,

            top_p=0.9,

            repetition_penalty=1.1,

            pad_token_id=tokenizer.pad_token_id,

            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    # Extract only the response
    if "Response:" in generated_text:

        response = generated_text.split(
            "Response:",
            1
        )[1].strip()

    else:

        response = generated_text

    return response


# =========================================================
# TITLE
# =========================================================

st.title("🤖 Multilingual NLP Application")

st.write(
    "A Streamlit application containing two"
    "Transformer models."
)

st.divider()


# =========================================================
# MODEL SELECTION
# =========================================================

model_choice = st.selectbox(
    "Select Model",
    [
        "mBERT - Natural Language Inference",
        "GPT-2 - Balochi Text Generation"
    ]
)


# =========================================================
# mBERT PAGE
# =========================================================

if model_choice == "mBERT - Natural Language Inference":

    st.header("🌍 mBERT - Natural Language Inference")

    st.write(
        "Compare two sentences and determine their relationship."
    )

    premise = st.text_area(
        "Premise",
        placeholder="Example: A boy is playing football."
    )

    hypothesis = st.text_area(
        "Hypothesis",
        placeholder="Example: A boy is playing a sport."
    )

    if st.button(
        "🔍 Predict",
        type="primary"
    ):

        if not premise.strip():

            st.warning("Please enter a premise.")

        elif not hypothesis.strip():

            st.warning("Please enter a hypothesis.")

        else:

            try:

                with st.spinner(
                    "Making prediction..."
                ):

                    label, confidence = predict_nli(
                        premise,
                        hypothesis
                    )

                st.success("Prediction completed.")

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Prediction",
                        label
                    )

                with col2:

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

            except Exception as error:

                st.error(
                    "mBERT prediction failed."
                )

                st.exception(error)


# =========================================================
# GPT-2 PAGE
# =========================================================

else:

    st.header("📝 GPT-2 - Balochi Text Generation")

    st.write(
        "Enter an instruction and GPT-2 will generate a response."
    )

    prompt = st.text_area(
        "Instruction",
        placeholder="Example: Explain the importance of education.",
        height=150
    )

    if st.button(
        "✨ Generate Response",
        type="primary"
    ):

        if not prompt.strip():

            st.warning(
                "Please enter an instruction."
            )

        else:

            try:

                with st.spinner(
                    "Generating response..."
                ):

                    response = generate_text(
                        prompt
                    )

                st.success(
                    "Response generated."
                )

                st.subheader(
                    "Generated Response"
                )

                st.write(response)

            except Exception as error:

                st.error(
                    "GPT-2 generation failed."
                )

                st.exception(error)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "mBERT + GPT-2 | "
    "Streamlit"
)