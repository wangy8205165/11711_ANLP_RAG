import requests
from bs4 import BeautifulSoup
import re, json, os
from urllib.parse import urljoin, urlparse

# import trafilatura, requests
# url = "https://trustarts.org/"
# html = requests.get(url).text
# # print(html)
# text = trafilatura.extract(html)
# print(text)


# url = "https://trustarts.org/"
# res = requests.get(url)
# soup = BeautifulSoup(res.text, "html.parser")
# text = soup.get_text(" ", strip=True)
# print(text)

#============= Configuration ===========================
entry = "https://trustarts.org/"
# entry = "https://carnegiemuseums.org/"
# entry = "https://www.heinzhistorycenter.org/"
# entry = "https://littleitalydays.com/"
# entry = "https://www.thefrickpittsburgh.org/"
# entry = "https://www.visitpittsburgh.com/events-festivals/food-festivals/"
# entry = "https://www.picklesburgh.com/"
# entry = "https://www.pghtacofest.com/"
# entry = "https://pittsburghrestaurantweek.com/"
# entry = "https://bananasplitfest.com/"
# entry = "https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/"
# entry = "https://www.mlb.com/pirates"
# entry = "https://www.steelers.com/"
# entry = "https://www.nhl.com/penguins/"

SEED_URL = entry    # url entry
DOMAIN = urlparse(SEED_URL).netloc  # ensure only access links within this website
OUT_FILE = "data_clean/chunks.jsonl"
OUT_TXT = "output_trust.txt"
CHUNK_SIZE = 300
OVERLAP = 50
MAX_PAGES = 30  # prevent scraping too much

visited = set()   # store the url that has been visited 
to_visit = [SEED_URL]  # url that's waiting to be visited
collected_texts = [] # store all the text obtained

open(OUT_TXT, "w").close() # Clear all the content in the txt file

def write_to_txt(text):
    with open(OUT_TXT, "a", encoding="utf-8") as f:
        f.write(text + "\n\n")


# ============ Step 1: Download the url ============
def fetch_page(url):
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json"}
    try:
        res = requests.get(url,headers=headers, timeout=10)
        res.encoding = res.apparent_encoding
        res.raise_for_status()
        return res.text
    except Exception as e:
        print(f"[ERROR] Failed {url}: {e}")
        return ""

# ============ Step 2: Extract the raw text and clean ============
def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.extract()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============ Step 3: Chunking ============
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks

# ============ Step 4: Save chunks to jsonl ============
def save_chunks(chunks, url, out_file, start_idx=0):
    with open(out_file, "a", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=start_idx):
            record = {
                "chunk_id": f"{url.replace('https://','').replace('http://','').replace('/','_')}_{i:04d}",
                "source": url,
                "text": chunk
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return start_idx + len(chunks)

# ============ Step 5: extract the links ============
def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])  # 绝对路径
        if urlparse(href).netloc == DOMAIN:  # 只保留本站链接
            if "#" in href:  # 去掉锚点
                href = href.split("#")[0]
            links.append(href)
    return links

# ============= Main Pipeline ===================
def main():
    # if os.path.exists(OUT_FILE):
    #     os.remove(OUT_FILE)
    # idx = 0
    pages_crawled = 0

    while to_visit and pages_crawled < MAX_PAGES:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"[INFO] Crawling {url}")
        html = fetch_page(url)
        if not html:
            continue
        
        text = clean_text(html)
        if len(text) < 50:
            print(f"[SKIP] Too short: {url}")
            continue
        if text in collected_texts:
            print(f"[SKIP] Duplicate content: {url}")
            continue 
        
        collected_texts.append(text)
        write_to_txt(text)
        

        print(f"Page{pages_crawled}")
        # print(text)
        # print("="*50)

        new_links = extract_links(html, url)
        for link in new_links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)
        
        pages_crawled += 1
    print(f"[DONE] Crawled {pages_crawled} pages, saved corpus to {OUT_FILE}")

if __name__ == "__main__":
    main()