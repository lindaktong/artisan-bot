import requests
from bs4 import BeautifulSoup
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from requests.exceptions import ConnectionError, Timeout
from openai import OpenAI
import tiktoken
import math
import pandas as pd
from urllib.parse import urljoin

client = OpenAI()
EMBEDDING_MODEL = 'text-embedding-3-large'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

# URL of the page
url = "https://support.artisan.co/en/collections/6492032-artisan-sales"

# Gets all linked pages from a single webpage
def fetch_all_links_from_page(url):
    # Fetch the content of the page
    response = requests.get(url)
    html_content = response.content
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    # Find all links to blog posts
    links = soup.find_all('a', href=True)
    # Extract and print the URLs
    blog_post_urls = []
    for link in links:
        href = link['href']
        # Check if the link is a relative URL, then prepend the base URL
        if not href.startswith('http'):
            href = url.rsplit('/', 1)[0] + '/' + href
        blog_post_urls.append(href)
    result_links = []
    # Save the blog post URLs
    for post_url in blog_post_urls:
        result_links.append(post_url)
    return result_links

def crawl_all_pages(base_url):
    visited = set()
    pages = set()
    to_visit = [base_url]
    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)
        try:
            response = requests.get(url)
            if response.status_code == 200:
                pages.add(url)
                soup = BeautifulSoup(response.content, "html.parser")
                for link in soup.find_all('a', href=True):
                    full_url = urljoin(base_url, link['href'])
                    if base_url in full_url and full_url not in visited:
                        to_visit.append(full_url)
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
    return sorted(pages)

# Get num tokens
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Vanilla embedding function
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    response = client.embeddings.create(input=text_or_tokens, model=model)
    return response.data[0].embedding

def chunk_tokens(text, encoding_name, chunk_length):
    chunked_tokens = []
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    num_chunks = math.ceil(len(tokens) / chunk_length)
    for i in range(num_chunks):
        start = i * chunk_length
        end = min((i + 1) * chunk_length, len(text))
        chunk = tokens[start:end]
        chunked_tokens.append(chunk)
    return chunked_tokens

def len_safe_get_embedding(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING, average=True):
    chunked_tokens = chunk_tokens(text, encoding_name=EMBEDDING_ENCODING, chunk_length=EMBEDDING_CTX_LENGTH)
    chunk_embeddings = client.embeddings.create(input=chunked_tokens, model=EMBEDDING_MODEL).data
    chunk_embeddings = [chunk.embedding for chunk in chunk_embeddings]
    chunk_lens = [len(chunk) for chunk in chunked_tokens]
    if average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2), retry=retry_if_exception_type((ConnectionError, Timeout)))
def robust_get(url):
    return requests.get(url)

def process_link(url):
    print(f"Processing URL: {url}")
    try:
        link_response = requests.get(url)
        if link_response.status_code == 200:
            html = link_response.text
            soup = BeautifulSoup(html, 'html.parser')
            # Extract the title
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else "No Title"
            # Extract all text from the page
            extracted_text = soup.get_text(separator=' ', strip=True)
            if extracted_text:
                # Prepend the title to the extracted content
                full_text = f"{title}\n\n{extracted_text}"
                print(f"Extracted Text: {full_text[:500]}...")  # Print the first 500 characters for preview
                return full_text, len_safe_get_embedding(full_text, model="text-embedding-3-small")
    except Exception as e:
        print(f"Failed to process URL {url}: {e}")
    return None, None

def fetch_and_process_pages(links):
    texts = []
    embeddings = []
    for link in links:
        text, embedding = process_link(link)
        if text and embedding:
            texts.append(text)
            embeddings.append(embedding)
    
    df = pd.DataFrame({"text": texts, "embedding": embeddings})
    return df

# Main execution
all_support_links = fetch_all_links_from_page('https://support.artisan.co/en/collections/6492032-artisan-sales')
all_links = crawl_all_pages('https://artisan.co')
print(all_links)
df = fetch_and_process_pages(all_support_links + all_links)

# df now contains the text and their corresponding embeddings
print(df.head())  # Display the first few rows of the DataFrame
SAVE_PATH = "artisan.csv"
df.to_csv(SAVE_PATH, index=False)

