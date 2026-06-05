import os
import requests
from bs4 import BeautifulSoup

def get_doc(url):
    headers = {
        "User-Agent": "AdelMultimodalRAG/1.0 (student project)"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Main WikiPedia article content
    content = soup.find('div', {'class': 'mw-parser-output'})
    if not content:
        print(f"Warning: No content found for URL: {url}")
        return
    else:
        print(f"Content found for URL: {url}")

    # Remove references, tables, and other non-essential elements
    for element in content.find_all(['sup', 'table', 'script', 'style']):
        element.decompose()
    if not content:
        print(f"Warning: Content has been washed out after decomposition")
        return
    else:
        print(f"Content still present after decomposition for URL")
    
    text = content.get_text(separator='\n')
    if not text:
        print(f"Warning: No text remaining for URL: {url}")
        return
    else:
        print(f"Text extracted successfully for URL: {url}")
    # Clean empty lines:
    #cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    # Get title of the article
    title = soup.find('h1', {'id': 'firstHeading'}).text.strip()

    os.makedirs("Docs", exist_ok=True)

    with open(os.path.join("Docs", title +'.txt'), 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Document for '{title}' has been saved successfully at Docs/{title}.txt")



