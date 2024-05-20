"""
This module contains necessary helper functions for dhq keyword and full-paper-based
paper recommendation.
"""

__author__ = "The Digital Humanities Quarterly Data Analytics Team"
__license__ = "MIT"
__version__ = "0.0.2"

import os
import re
from bs4 import Tag
from typing import Union
from typing import Dict, List, Optional

from bs4 import BeautifulSoup


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
    title = remove_excessive_space(soup.find("title").text)

    # extract publication year, volume, and issue
    publication_date = soup.find("date", {"when": True})
    if publication_date:
        publication_year = publication_date["when"][:4]
    else:
        raise RuntimeError(f'{paper_id} does not have publication year in xml.')

    volume = (
        soup.find("idno", {"type": "volume"}).text
        if soup.find("idno", {"type": "volume"})
        else ''
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
        if getattr(child, 'name', None) is None:
            text += child.string.strip() + " "
        else:
            text += extract_text_recursive(child)
    return remove_excessive_space(text)
