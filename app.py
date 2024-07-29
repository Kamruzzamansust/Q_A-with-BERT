import streamlit as st
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model = BertForQuestionAnswering.from_pretrained('./bert_qa_model/')
    tokenizer = BertTokenizer.from_pretrained('./bert_qa_model/')
    return model, tokenizer

model, tokenizer = load_model()

def get_answer(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt')
    input_ids = inputs['input_ids'].tolist()[0]

    # Get the model scores
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Get the most likely beginning and end of answer with the argmax of the score
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer

st.title("Question Answering with BERT")

question = st.text_input("Enter your question:")
context = st.text_area("Enter the context:")

if st.button("Get Answer"):
    if question and context:
        answer = get_answer(question, context)
        st.write("Answer:", answer)
    else:
        st.write("Please provide both question and context.")
