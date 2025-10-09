'''
Instead of crawling website one by one, this program aims to crawl all the website in a single run
'''

from tkinter.tix import MAX
import requests
from bs4 import BeautifulSoup
import re, json, os
from urllib.parse import urljoin, urlparse

# from torch import chunk

wikipedia_url = "https://en.wikipedia.org/wiki/Pittsburgh"
wikipediahistory_url = "https://en.wikipedia.org/wiki/Pittsburgh"
pitts_gov_url = "https://www.pittsburghpa.gov/Home"
britannica_url = "https://www.britannica.com/place/Pittsburgh"
visitpitts_url = "https://www.visitpittsburgh.com/"
pitts_tax_url = "https://www.pittsburghpa.gov/City-Government/Finance-Budget/Taxes/Tax-Forms" # This is all PDF files
cmu_url = "https://www.cmu.edu/about"
pitts_event = "https://pittsburgh.events/"
downtown_pitts = "https://downtownpittsburgh.com/events/"
pghcitypaper_url = "https://community.pghcitypaper.com/pittsburgh/EventSearch?v=d"
event_cmu_url = "https://events.cmu.edu/"
event_cmu_url2 = "https://www.cmu.edu/engage/events"
pittssymphony_url = "https://www.pittsburghsymphony.org/"
pittsopera_url = "https://pittsburghopera.org/"
trustarts_url = "https://trustarts.org/"
carnegie_museum_url = "https://carnegiemuseums.org/"
heinzhistory_url = "https://www.heinzhistorycenter.org/"
thefrick_url = "https://www.thefrickpittsburgh.org/"
visitpitts_festival_url = "https://www.visitpittsburgh.com/events-festivals/food-festivals/"
pickleburgh_url = "https://www.picklesburgh.com/"
pghtacofest_url = "https://www.pghtacofest.com/"
pittsresweek_url = "https://pittsburghrestaurantweek.com/"
littleitaly_url = "https://littleitalydays.com/"
visitpitts_todo_url = "https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/"
banana_url = "https://bananasplitfest.com/"
mlb_url = "https://www.mlb.com/pirates"
nhl_url = "https://www.nhl.com/penguins/"
steeler_url = "https://www.steelers.com/"

# txt_names = ["wiki.txt", "wiki_his.txt", "pittsgov.txt",
#              "britannica.txt", "visitpitts.txt","cmu.txt",
#              "pittseven.txt", "downtown_pitts.txt","pghcitypaper.txt",
#              "event_cmu.txt","event_cmu2.txt","pitsymphony.txt"
#              ]


# URL_LIST = [trustarts_url, carnegie_museum_url,heinzhistory_url,thefrick_url,visitpitts_festival_url,pickleburgh_url,pghtacofest_url]

URL_LIST = [visitpitts_festival_url]
MAX_DEPTH = 2
MAX_PAGES = 50   # How many subpages to crawl
CHUNK_SIZE = 500
OVERLAP = 50
CHUNK_MIN_LEN = 50  # The minimum length of chunks to accpet
# OUT_JSON = "data/chunks_trustarts.jsonl"
OUT_FILE_TXT = "raw_text/visitpitts_festival.txt"

# ============ Step 1: Download the Website ============
def fetch_page(url):
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        res.encoding = res.apparent_encoding
        ctype = res.headers.get("Content-Type", "")
        if "text" not in ctype and "html" not in ctype:
            return ""
        return res.text
    except Exception as e:
        print(f"[ERROR] Failed {url}: {e}")
        return ""
    
# ============ Step 2: Clean the raw text ============
def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.extract()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ============ Step 3: extract the links of subpages ============
def extract_links(html, base_url, domain):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])  # absolute path
        if urlparse(href).netloc == domain:  # Only keep the links within the domain
            if "#" in href: 
                href = href.split("#")[0]
            links.append(href)
    return links

# ============ Chunk the text =========================
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks

# ========= Save the chunks to json file =========
def save_chunks(chunks, source, counter):
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "a", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "chunk_id": f"{counter:04d}",
                "source": source,
                "text": chunk
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            counter += 1
    return counter


# ============ Step 4: Crawl a single website ============
def crawl_site(seed_url,counter):
    # domain = urlparse(seed_url).netloc.replace("www.", "")
    domain = urlparse(seed_url).netloc
    # file_name = domain.split(".")[1]
    out_file = OUT_FILE_TXT
    open(out_file, "w").close() # Clear all the content in the txt file
    print(f"\n[START] Crawling site: {domain}")
    print(f"[INFO] Output file: {OUT_FILE_TXT}\n")

    visited =set ()
    to_visit = [(seed_url, 1)]



    # visited = set()
    # to_visit = [seed_url]
    collected_texts = []

    pages_crawled = 0
    while to_visit and pages_crawled < MAX_PAGES:
        url,depth = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"[INFO] Crawling {url}")
        html = fetch_page(url)
        if not html:
            continue

        text = clean_text(html)
        if len(text) < CHUNK_MIN_LEN:
            continue
        if text in collected_texts:  # Skip is content is repeated
            print(f"[SKIP] Duplicate: {url}")
            continue
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(text + "\n\n")

        collected_texts.append(text)
        pages_crawled += 1
        print(f"[SAVE] ({pages_crawled}) {url}")

        # chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
        # counter = save_chunks(chunks, url, counter)

        # extract the new links 
        if depth < MAX_DEPTH:
            new_links = extract_links(html, url, domain)
            # print(f"new links: {new_links}")
            for link in new_links:
                if link not in visited and all(link != l for l, _ in to_visit):
                    to_visit.append((link, depth+1))

    print(f"[DONE] {domain} -> Crawled {pages_crawled} unique pages.\n")

# ============ Step 5: main pipeline ============
def main():
    # if os.path.exists(OUT_JSON):
    #     os.remove(OUT_JSON)  # Clear the old data
    counter = 0
    for site in URL_LIST:
        crawl_site(site,counter)
    print("ALL DONE!")

if __name__ == "__main__":
    main()