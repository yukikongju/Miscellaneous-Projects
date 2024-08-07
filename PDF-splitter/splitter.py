import json
import os

from PyPDF2 import PdfFileWriter, PdfFileReader

def read_json(json_file):
    """ 
    Function that read json file 
    """
    with open(json_file) as f:
        data = json.load(f)
    return data
    

def create_subpdf(data):
    """ 
    Function that create pdf from another pdf starting at page {start_page} and 
    ending at {end_page}

    Parameters
    ----------
    data: json dict

    Returns
    -------

    Examples
    --------
    
    """
    pdf_file = data['pdf_file']
    output_dir = data['output_dir']
    chapters = data['chapters']

    # initialize page scaler
    #  scaler = Transformation().scale(sx=0.6, sy=0.7)

    # create output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # read pdf 
    with open(pdf_file, 'rb') as pdf_file:
        # initialize pdf reader and writer
        reader = PdfFileReader(pdf_file, strict = False)

        #  split pdf into sub pdfs
        for chapter in chapters:
            chapter_name = chapter['name']
            start_page = chapter['start_page']
            end_page = chapter['end_page']
            writer = PdfFileWriter()
            for i in range(start_page, end_page):
                page = reader.getPage(i)
                #  page.addTransformation(scaler)
                writer.addPage(page)
            subpdf_name = f"{output_dir}/{chapter_name}.pdf"
            with open(subpdf_name, 'wb') as stream:
                writer.write(stream)
                stream.close()


def main():
    # read json file
    json_file = "PDF-splitter/configs/computer_vision_stanford.json"
    #  json_file = "PDF-splitter/configs/crafting_interpreters.json"
    #  json_file = "PDF-splitter/configs/grokking_deep_learning.json"
    #  json_file = "PDF-splitter/configs/pevzner_bioinformaticsI.json"
    #  json_file = "PDF-splitter/configs/pevzner_bioinformaticsII.json"
    data = read_json(json_file)

    # create subpdfs
    create_subpdf(data)
    

if __name__ == "__main__":
    main()
