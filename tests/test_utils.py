import unittest
from utils import extract_relevant_elements
from unittest.mock import patch, mock_open


class TestExtractRelevantElements(unittest.TestCase):

    def setUp(self):
        self.xml_data_1 = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0" xmlns:dhq="http://www.digitalhumanities.org/ns/dhq">
            <teiHeader>
                <fileDesc>
                    <titleStmt>
                        <title type="article" xml:lang="en">Response to <title rend="quotes">This Is Good</title> and Hello World</title>
                    </titleStmt>
                    <publicationStmt>
                        <idno type="DHQarticle-id">000664</idno>
                        <idno type="volume">016</idno>
                        <idno type="issue">4</idno>
                        <date when="2022-12-23">23 December 2022</date>
                    </publicationStmt>
                </fileDesc>
            </teiHeader>
        </TEI>"""

        self.xml_data_2 = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0" xmlns:dhq="http://www.digitalhumanities.org/ns/dhq">
            <teiHeader>
                <fileDesc>
                    <titleStmt>
                        <title type="article" xml:lang="en">Response to <title rend="quotes">This Is Good</title> and <title rend="quotes">That Is Bad</title></title>
                    </titleStmt>
                    <publicationStmt>
                        <idno type="DHQarticle-id">000666</idno>
                        <idno type="volume">016</idno>
                        <idno type="issue">4</idno>
                        <date when="2022-12-23">23 December 2022</date>
                    </publicationStmt>
                </fileDesc>
            </teiHeader>
        </TEI>"""

    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_case_1(self, mock_file):
        mock_file.return_value.read.return_value = self.xml_data_1
        result = extract_relevant_elements("dummy_path")
        self.assertEqual(result["title"],
                         'Response to "This Is Good" and Hello World')

    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_case_2(self, mock_file):
        mock_file.return_value.read.return_value = self.xml_data_2
        result = extract_relevant_elements("dummy_path")
        self.assertEqual(result["title"],
                         'Response to "This Is Good" and "That Is Bad"')


if __name__ == "__main__":
    unittest.main()
