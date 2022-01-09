import streamlit as st
import base64
from transformers import AutoTokenizer
import torch

filename = "t5-Arabic.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

# model = torch.load(filename, map_location=device)
model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

def summarizeText(text):
    text_encoding = tokenizer(
        text,
        max_length=250,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    generated_ids = model.generate(
        input_ids=text_encoding['input_ids'].to(device),
        attention_mask=text_encoding['attention_mask'].to(device),
        max_length=15,
        num_beams=4,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )    

    preds = [
            tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
    ]
    return "".join(preds)

def main():
    st.title("Text Summarization")
    st.markdown(
        f"""
            <style>
                .reportview-container .main .block-container{{
                    max-width: 1200px;
                    padding-top: 3rem;
                    padding-right: 1rem;
                    padding-left: 1rem;
                    padding-bottom: 5rem;
                }}
            </style>
            """,
        unsafe_allow_html=True,
    )
    post = st.text_area("",placeholder="Type Here", height=200)
    mechanism = st.radio(
        "Choose the mechanism : ",
        ('Text Rank', 'Machine learning'))
    
    if st.button("Summarize"):
        if mechanism == 'Machine learning':
            result=summarizeText(post)
        st.write(result)


if __name__=='__main__':
    main()
