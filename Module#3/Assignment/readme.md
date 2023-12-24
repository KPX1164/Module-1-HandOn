Run the script using the following command:
```python job_search.py <keyword> <data_file_path> --method <search_method>```
Replace <keyword> with the search term, <data_file_path> with the path to your CSV file, and <search_method> with either 'tfidf' or 'bm25'.

For example:
```python job_search.py python data/jobs.csv --method bm25```
