from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

def openPdf2Text(filepath:str) -> str:
    ''' Function to open pdf and return extracted text'''
    
    output_string = StringIO()
    with open(filepath, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
    return output_string.getvalue()

def openMultiPdf(list_files:str, dir_documents:str) -> list:
    ''' Function to open multiple pdf files with openPdf2Text function and return a list of strings '''
    
    dataset = [openPdf2Text(f'{dir_documents}{f}') for f in list_files]
    return dataset