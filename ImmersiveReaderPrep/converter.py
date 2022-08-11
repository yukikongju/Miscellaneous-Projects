#!/usr/bin/env python

import os
import glob
#  import aspose.words as aw

from docx import Document
from pdf2docx import Converter, parse
from PyPDF2 import PdfFileWriter, PdfFileReader

class Reader(object):


    def __init__(self, filepath, dirpath):
        self.filepath = filepath
        self.dirpath = dirpath

        os.makedirs(dirpath, exist_ok=True)
        
    def _splitPDF(self, step):
        """ step: num of pages for interval """
        reader = PdfFileReader(open(self.filepath, 'rb'))

        num_pages = reader.getNumPages()

        # create page range
        start = 1
        intervals = []
        while start < num_pages:
            end = start + step if (start + step < num_pages) else num_pages
            intervals.append((start, end))
            start += step + 1

        #  create word document every 20 pages
        for interval in intervals: 
            writer = PdfFileWriter()
            start, end = interval[0], interval[1]
            text = []

            for i in range(start, end):
                page = reader.getPage(i)
                writer.addPage(page)

            #  save file
            with open(f'{self.dirpath}/Page_{start}_{end}.pdf', 'wb') as stream:
            #  with open(f'{self.dirpath}/Page_{start}_{end}.docx', 'wb') as stream:
                writer.write(stream)
                stream.close()

    def convertAllPdfToWord(self):
        # 1. split PDF into smaller PDF files
        self._splitPDF(step=20)

        # 2. create docx file for pdf if doesnt exist (formatting wrong)
        #  for pdf_file in glob.glob(f'{self.dirpath}/*.pdf'):
        #      doc_file = pdf_file.replace('.pdf', '.docx')
        #      if not os.path.isfile(f'{self.dirpath}/{doc_file}'):
        #          self._convertPdfToWord(pdf_file, doc_file)

        
    def _convertPdfToWord(self, pdf_file, doc_file):
        converter = Converter(pdf_file)
        #  converter.convert(doc_file, start=0, end=None)
        converter.convert(doc_file)
        converter.close()
        #  parse(pdf_file, doc_file, start=0, end=None)


#  def test_aspose(pdffile):
#      doc = aw.Document(pdffile)
#      doc.save(pdffile.replace('.pdf', '.docx'))


if __name__ == "__main__":
    dirpath  = 'ImmersiveReaderPrep/ImmersiveReader'
    filepath = 'ImmersiveReaderPrep/F.pdf'
    #  splitter = Splitter(filepath, dirpath)
    #  splitter.splitPDF()
    reader = Reader(filepath, dirpath)
    reader.convertAllPdfToWord()

    # test aspose

    #  test_aspose(filepath)


        
