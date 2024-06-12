"""
This module contains necessary helper functions for dhq keyword and full-paper-based
paper recommendation.
"""

__author__ = "The Digital Humanities Quarterly Data Analytics Team"
__license__ = "MIT"
__version__ = "0.0.5"

import csv
import os
import re
from typing import Dict, List, Optional, Tuple, Union

from bs4 import BeautifulSoup, NavigableString, Tag

# some papers should not have recommendations, e.g., remembrance pieces
INGORE_ARTICLES = ["dhq-journal/articles/000493"]

# tsv files
BM25_TSV_PATH = "dhq-recs-zfill-bm25.tsv"
KWD_TSV_PATH = "dhq-recs-zfill-kwd.tsv"
SPCTR_TSV_PATH = "dhq-recs-zfill-spctr.tsv"


def get_articles_in_editorial_process(
    toc_xml: str = "dhq-journal/toc/toc.xml",
) -> List[str]:
    """
    Retrieve article IDs that are still in the editorial process from the toc.xml file.
    These articles should be removed from the recommendation system, as per Julia's
    guidance.

    Args:
        toc_xml: Path to the toc.xml file.

    Returns:
        A list of article IDs still in the editorial process.
    """
    with open(toc_xml, "r", encoding="utf-8") as file:
        xml_content = file.read()
    soup = BeautifulSoup(xml_content, "xml")

    # find all <journal editorial="true"> tags
    editorial_journals = soup.find_all("journal", editorial="true")
    article_ids = [
        item["id"]
        for journal in editorial_journals
        for item in journal.find_all("item")
    ]

    return article_ids


def extract_article_folders(directory: str) -> List[str]:
    """
    Extract folders that match DHQ article folder naming pattern, excluding the example
    article '000000'.

    Args:
        directory: The directory path to scan for article folders (in the cloned DHQ
            repo).

    Returns:
        A list of paths to folders that match the DHQ article naming convention.
    """
    fld_reg = re.compile(r"^00\d{4}$")
    filtered_folders = [
        entry.path
        for entry in os.scandir(directory)
        if entry.is_dir() and re.match(fld_reg, entry.name) and entry.name != "000000"
    ]

    return filtered_folders


def remove_excessive_space(t: str) -> str:
    """
    Remove redundant space and Zero Width Space from extracted text.
    Args:
        t: an extracted field

    Returns:
        a string properly spaced
    """
    # \u200B is the unicode for Zero Width Space
    t = t.replace("\u200B", "")
    t = re.sub(r"\s+", " ", t)

    return t


def extract_relevant_elements(xml_folder: str) -> Dict[str, Optional[str]]:
    """
    Extract relevant elements from a DHQ article XML file, including paper_id, authors
    (pipe concatenated if multiple), affiliations (pipe concatenated if multiple),
    title, abstract, url (heuristically construed with volume/issue/id), volume, issue,
    and dhq_keywords.

    Args:
        xml_folder: A path to a DHQ article XML file
        (e.g., 'dhq-journal/articles/000275').

    Returns:
        A dictionary containing the extracted information.
    """
    paper_id = xml_folder.split("/").pop()
    article_path = os.path.join(xml_folder, f"{paper_id}.xml")

    with open(article_path, "r", encoding="utf-8") as file:
        xml = file.read()
        soup = BeautifulSoup(xml, "xml")

    # extract title
    title_tag = soup.find("title")
    if title_tag:
        quotes_title_tag = title_tag.find("title", {"rend": "quotes"})
        if quotes_title_tag:
            # extract text before the quoted title
            before_quotes = "".join(
                str(content)
                for content in title_tag.contents
                if isinstance(content, NavigableString)
            )
            before_quotes = before_quotes.split(str(quotes_title_tag))[0].strip()
            before_quotes = remove_excessive_space(before_quotes)

            # extract the quoted title
            quotes = f' "{remove_excessive_space(quotes_title_tag.text)}"'

            # extract text after the quoted title
            after_quotes = "".join(
                str(content)
                for content in title_tag.contents
                if isinstance(content, NavigableString)
                and content not in quotes_title_tag.contents
            )
            after_quotes = after_quotes.split(str(quotes_title_tag))[-1].strip()
            after_quotes = remove_excessive_space(after_quotes)

            title = before_quotes + quotes + " " + after_quotes
        else:
            title = remove_excessive_space(title_tag.text)
    else:
        title = ""

    # extract publication year, volume, and issue
    publication_date = soup.find("date", {"when": True})
    if publication_date:
        publication_year = publication_date["when"][:4]
    else:
        publication_year = ""

    volume = (
        soup.find("idno", {"type": "volume"}).text
        if soup.find("idno", {"type": "volume"})
        else ""
    )
    # trim leading 0s
    volume = volume.lstrip("0")
    issue = (
        soup.find("idno", {"type": "issue"}).text
        if soup.find("idno", {"type": "issue"})
        else None
    )

    # heuristically construct url using volume, issue, and paper_id
    url = (
        f"https://digitalhumanities.org/dhq/vol/"
        f"{volume}/{issue}/{paper_id}/{paper_id}.html"
    )

    # extract authors and affiliations
    authors_tag = []
    affiliations_tag = []
    for author_info in soup.find_all("dhq:authorInfo"):
        author_name_tag = author_info.find("dhq:author_name")
        if author_name_tag:
            # extract the full name as text, including proper spacing
            full_name = " ".join(author_name_tag.stripped_strings)
            authors_tag.append(full_name)
        else:
            authors_tag.append("")
        affiliation_tag = author_info.find("dhq:affiliation")
        affiliation = affiliation_tag.get_text(strip=True) if affiliation_tag else ""
        affiliations_tag.append(affiliation)

    authors = " | ".join([remove_excessive_space(name) for name in authors_tag])
    affiliations = " | ".join([remove_excessive_space(aff) for aff in affiliations_tag])

    # extract abstract
    abstract_tag = soup.find("dhq:abstract")
    if abstract_tag:
        paragraphs = abstract_tag.find_all("p")
        abstract = " ".join(p.get_text(strip=True) for p in paragraphs)
        abstract = remove_excessive_space(abstract)
    else:
        abstract = ""

    # extract DHQ keywords
    dhq_keywords_tags = soup.find_all("term", {"corresp": True})
    dhq_keywords = (
        [term["corresp"].lower().strip() for term in dhq_keywords_tags]
        if dhq_keywords_tags
        else [""]
    )

    # extract full body text including headings
    body_text = ""
    body = soup.find("body")
    if body:
        # iterate through all elements, recursively extracting text including headings
        body_text = extract_text_recursive(body)

    return {
        "paper_id": paper_id,
        "title": title.strip(),
        "volume": volume,
        "issue": issue,
        "publication_year": publication_year.strip(),
        "authors": authors.strip(),
        "affiliations": affiliations.strip(),
        "abstract": abstract.strip(),
        "url": url,
        "dhq_keywords": dhq_keywords,
        "body_text": body_text.strip(),
    }


def extract_text_recursive(element: Union[Tag, BeautifulSoup]) -> str:
    """
    Recursively extract text from an element, including nested tags.

    Args:
        element: The BeautifulSoup tag element to extract text from.

    Returns:
        A concatenated string of all text within the element.
    """
    text = ""
    for child in element.children:
        if getattr(child, "name", None) is None:
            text += child.string.strip() + " "
        else:
            text += extract_text_recursive(child)
    return remove_excessive_space(text)


def validate_metadata(metadata: list) -> Tuple[list, list]:
    """
    Validate metadata for any fields with zero length and filter out such entries.

    Args:
        metadata: A list of dictionaries containing metadata for articles.

    Returns:
        A tuple containing two lists:
        - The first list contains the valid metadata entries (useful for subsequent
            computation).
        - The second list contains the valid metadata entries with non-zero length
            fields (which prefills fields in the final tsv table).
    """
    indices = []
    recs = []
    for index, m in enumerate(metadata):
        # pick up the tsv naming convention
        rec = {
            "Article ID": m["paper_id"],
            "Pub. Year": m["publication_year"],
            "Authors": m["authors"],
            "Affiliations": m["affiliations"],
            "Title": m["title"],
            "url": m["url"],
        }
        # check for 0-length text and print the corresponding key
        has_zero_length_value = False
        for key, value in rec.items():
            if value == "":
                # don't filter out articles written by a corporate author using their
                # affiliation name as the author name and reasonably leave the
                # affiliation field blank
                if key == "Affiliations" and rec["Authors"]:
                    print(
                        f"{m['paper_id']} seems to be a corporate author because its "
                        f"{key} is missing but {rec['Authors']=} isn't. "
                        f"Will be included in the recommendations."
                    )
                else:
                    print(
                        f"{m['paper_id']}'s {key} is missing. "
                        f"Will not be included in the recommendations."
                    )
                    has_zero_length_value = True
        if not has_zero_length_value:
            indices.append(index)
            recs.append(rec)
    valid_metadata = [m for i, m in enumerate(metadata) if i in indices]
    return valid_metadata, recs


def get_metadata() -> List[Dict]:
    """
    Get metadata from raw xml files.
    Notice, we will filter out articles:
        - in editorial process and
        - otherwise specified.
    Returns:
        Metadata for each article.
    """
    # get all xml files
    xml_folders = extract_article_folders("dhq-journal/articles")
    # ignore articles in editorial process (should not be considered in recommendation)
    xml_to_ignore = [
        os.path.join("dhq-journal/articles", f)
        for f in get_articles_in_editorial_process()
    ]
    # ignore papers otherwise specified
    xml_to_ignore.extend(INGORE_ARTICLES)
    xml_folders = [f for f in xml_folders if f not in xml_to_ignore]

    metadata = []
    for xml_folder in xml_folders:
        paper_id = xml_folder.split("/").pop()
        paper_path = os.path.join(xml_folder, f"{paper_id}.xml")
        if os.path.exists(paper_path):
            metadata.append(extract_relevant_elements(xml_folder))
    return metadata


def sort_then_save(recs: List[Dict[str, str]], tsv_path: str) -> None:
    """
    Sorts a list of article recommendations by 'Article ID' and saves them to a TSV
    file.

    Args:
        recs: A list of dictionaries, each representing fields useful in downstream
            article recommendation system.
        tsv_path: The file path where the sorted recommendations TSV will be saved.

    Returns:
        None
    """
    # sort and save
    recs = sorted(recs, key=lambda x: x["Article ID"])
    header = list(recs[0].keys())
    # move 'url' to the end to follow naming conventions
    header.append(header.pop(header.index("url")))

    with open(tsv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in recs:
            writer.writerow(row)
