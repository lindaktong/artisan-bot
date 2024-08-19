from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import ast
from openai import OpenAI
import pandas as pd
import tiktoken
import os
from scipy import spatial

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI client
client = OpenAI()

# Models and constants
EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4o"

# Load pre-chunked text and pre-computed embeddings
df = pd.read_csv("artisan.csv")
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# Initialize a list to store the last 10 user messages
last_10_messages = []

# Helper functions 
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def strings_ranked_by_relatedness(query: str, df: pd.DataFrame, relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y), top_n: int = 100):
    query_embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [(row["text"], relatedness_fn(query_embedding, row["embedding"])) for i, row in df.iterrows()]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def query_message(query: str, df: pd.DataFrame, model: str, token_budget: int) -> str:
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below articles on Artisan to answer the subsequent question."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nArtisan article:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    return message + question

def add_message_to_history(message: str, history: list):
    history.append(message)
    if len(history) > 10:
        history.pop(0)

def construct_message_context(history: list, current_message: str) -> list:
    messages = [{"role": "system", "content": "You are Ava, Artisan's AI BDR. You use the provided sources to answer user questions about the product in a clear, helpful, and friendly manner. The user won't see the retrieved sources, so DON'T use phrases like 'according to the articles' or 'based on the provided articles.'"}]
    for past_message in history:
        messages.append({"role": "user", "content": past_message})
    messages.append({"role": "user", "content": current_message})
    return messages

def manage_token_count(messages: list, model: str = GPT_MODEL, max_tokens: int = 4096 - 500):
    while num_tokens(' '.join([m['content'] for m in messages]), model=model) > max_tokens:
        if len(messages) > 2:
            messages.pop(1)
        else:
            break
    return messages

def ask(query: str, df: pd.DataFrame = df, model: str = GPT_MODEL, token_budget: int = 4096 - 500, history: list = last_10_messages) -> str:
    message = query_message(query, df, model=model, token_budget=token_budget)
    add_message_to_history(query, history)
    messages = construct_message_context(history, message)
    messages = manage_token_count(messages, model=model, max_tokens=token_budget)
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    response_message = response.choices[0].message.content
    return response_message

# Request model for FastAPI
class ChatRequest(BaseModel):
    message: str

# FastAPI endpoint
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = ask(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))