from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import os
import shutil


question = "Where was the Football League founded?"
reference_text = "In 1888, The Football League was founded in England, becoming many professional football competitions. During the 20th century, several football leagues grew to become some of the most popular team sports in the world."


from transformers import BertForQuestionAnswering, BertTokenizer
import torch

def get_answer_using_bert(question, reference_text):
    # Load pretrained model for Question Answering
    bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    
    # Load Vocabulary
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    
    # Perform tokenization on input text
    input_ids = bert_tokenizer.encode(question, reference_text)
    input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)
    
    # Find index of first occurrence of [SEP] token
    sep_location = input_ids.index(bert_tokenizer.sep_token_id)
    first_seg_len, second_seg_len = sep_location + 1, len(input_ids) - (sep_location + 1)
    seg_embedding = [0] * first_seg_len + [1] * second_seg_len
    
    # Test model on our example
    model_scores = bert_model(
        torch.tensor([input_ids]),
        token_type_ids=torch.tensor([seg_embedding])
    )
    
    # Find the start and end locations of the answer
    ans_start_loc = torch.argmax(model_scores[0])
    ans_end_loc = torch.argmax(model_scores[1])
    
    # Extract the tokens corresponding to the answer
    result = ' '.join(input_tokens[ans_start_loc:ans_end_loc + 1])
    result = result.replace(' ##', '')
    
    
    return result

bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Define the output directory
output_dir = './bert_qa_model/'

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the model and tokenizer
bert_model.save_pretrained(output_dir)
bert_tokenizer.save_pretrained(output_dir)

# Create a zip file of the saved model directory
shutil.make_archive('bert_qa_model', 'zip', output_dir)

print("Model saved and compressed successfully.")