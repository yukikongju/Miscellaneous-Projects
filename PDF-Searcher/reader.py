#!/env/bin/python

#  import textract
#  import PyPDF2
#  from tika import parser


class Reader(object):

    def __init__(self, filename):
        self.filename = filename

    def __str__(self):
        return f'filename: {self.filename}'

    def read_file(self):
        pass

class PDFReader(Reader):

    def __init__(self, filename):
        super().__init__(filename)

    def read_file(self):
        pass
        

def main():
    reader = PDFReader("PDF-Searcher/devoir1.pdf")
    reader.read_file()
    

if __name__ == "__main__":
    main()

