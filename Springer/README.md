# Springer Download 

Download excel sheet with all springer books in series

## Usage 

##### Download book series as excel file

Find series identification number

```python
    springer = SpringerBookSeries(3464)
    springer.save_series_as_csv()
```

##### Download book infor as excel file

Find book url

```python
    book = SpringerBook('https://link.springer.com/book/10.1007/978-3-030-44074-9')
    springer.download_book_info_as_excel()
```

## ToDos

- [ ] create excel file for individual book
- [ ] Download all books in book series

## Ressources

- [Springer Nature](https://www.springernature.com/gp/booksellers/catalogs-title-selections)

