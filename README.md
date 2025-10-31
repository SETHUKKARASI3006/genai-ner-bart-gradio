## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Current methods for text analysis often struggle with accurately and efficiently extracting contextual information like Named Entities (e.g., person names, locations, organizations) from unstructured text, particularly when dealing with complex linguistic structures. The objective is to implement a robust Transformer-based model (specifically a BERT-based model is used in the code, standing in for the high-performance of a fine-tuned BART/Transformer model) for the NER task, and package it into an intuitive, shareable web application using Gradio. This prototype must demonstrate the capability to correctly identify and categorize entities and merge fragmented word tokens for clean output visualization.

### DESIGN STEPS:

#### STEP 1:
Choose a fine-tuned BART/BERT model for NER and obtain its Hugging Face API endpoint URL.

#### STEP 2:
Load API key securely and create a helper function for communication with the Hugging Face Inference API.

#### STEP 3:
Implement the merge_tokens function to combine sub-word tokens into single, coherent entities.

#### STEP 4:
Define the main ner function to call the API, process the output, and run the token merging logic.

#### STEP 5:
Set up the gr.Interface with a gr.Textbox input and a gr.HighlightedText output for visualization.

#### STEP 6:
Define the title, description, and examples for the Gradio app to enhance user experience.

#### STEP 7:
Deploy the prototype using demo.launch() for real-time testing and evaluation.

### PROGRAM:

```
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Helper function
import requests, json

#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))

text = ('''Generative AI (GenAI) is a powerful type of artificial intelligence capable of creating new 
        content—including text, images, code, and music—rather than just classifying or analyzing 
        existing data. At its core are Large Language Models (LLMs) and other deep learning 
        architectures trained on vast datasets. These models learn patterns and structures, allowing 
        them to produce outputs that are remarkably original and contextually relevant. The applications 
        of GenAI span industries, from automating content creation and accelerating scientific discovery 
        to revolutionizing software development and personalized customer experiences. Its rapid 
        evolution marks a significant shift, raising important questions about creativity, ethics, and the 
        future of work.''')

get_completion(text)

import gradio as gr
def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']
    
gr.close_all()
demo = gr.Interface(fn=summarize, inputs="text", outputs="text")
demo.launch(share=True, server_port=int(os.environ['PORT1']))

import gradio as gr

def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']

gr.close_all()
demo = gr.Interface(fn=summarize, 
                    inputs=[gr.Textbox(label="Text to summarize", lines=6)],
                    outputs=[gr.Textbox(label="Result", lines=3)],
                    title="Text summarization with distilbart-cnn",
                    description="Summarize any text using the `shleifer/distilbart-cnn-12-6` model under the hood!"
                   )
demo.launch(share=True, server_port=int(os.environ['PORT2']))

API_URL = os.environ['HF_API_NER_BASE'] #NER endpoint
text = "My name is Sethukkarasi, I'm building DeepLearningAI and I live in Chennai"
get_completion(text, parameters=None, ENDPOINT_URL= API_URL)

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    return {"text": input, "entities": output}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    #Here we introduce a new tag, examples, easy to use examples for your application
                    examples=["My name is Sethukkarasi and I live in Chennai", "My name is Elan and work at Pondicherry"])
demo.launch(share=True, server_port=int(os.environ['PORT3']))

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["My name is Sethukkarasi, I'm building DeeplearningAI and I live in Chennai", "My name is Elan, I live in Pondicherry and work at HuggingFace"])

demo.launch(share=True, server_port=int(os.environ['PORT4']))

gr.close_all()
```

### OUTPUT:

<img width="479" height="531" alt="image" src="https://github.com/user-attachments/assets/2fa61bf2-82a5-42bb-8686-479e1c248b56" />

<img width="493" height="541" alt="image" src="https://github.com/user-attachments/assets/8bb672f5-5e5d-4f51-a981-b50723e62629" />

<img width="493" height="445" alt="image" src="https://github.com/user-attachments/assets/f93835e1-ab6f-453c-a02c-02e52f1a2bb2" />


### RESULT:
Thus, a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation is designed and developed successfully.
