# Instalation

```
pip install pip install automlhelper
```

# Usage

See [example.py](https://github.com/N2BBrasil/automl-helper/blob/master/example.py)


## AutoMLHelper
```python
AutoMLHelper(self, bucket_name, csv_path, project_name)
```
AutoML Helper

Args:
    bucket_name (int): The name of the bucket utilized in AutoML e.g.: my-bucket-vcm
    csv_path (str): The path where yours CSVs with reference of images and classes of your images. e.g.: csv/
    project_name (str): Your Google Cloud project

### tag_performance
```python
AutoMLHelper.tag_performance(self)
```
Returns tag `display_name` and performance object.

Warning: This connection between tag display_name and the performance object is not returned by Google`s API, this method utilizes an alternative way that can stop working at any moment.

Returns:
    list: (display_name, performance_object)

### tag_and_performance
```python
AutoMLHelper.tag_and_performance(self, key='base_au_prc')
```
Returns tag `display_name` and performance metric.

Args:
    key (str): performance metric of performance object defaults: 'base_au_prc'

Returns:
    list: (display_name, performance_metric)

### best_performing
```python
AutoMLHelper.best_performing(self, key='base_au_prc', limit=0, reverse=True)
```
Returns best or worst tags by performance metric.

Args:
    key (str): performance metric of the performance object defaults: 'base_au_prc'
    limit (int): max number of results
    reverse (bool): True for best performing, False for worst performing

Returns:
    list: (display_name, performance_metric)

### get_tags
```python
AutoMLHelper.get_tags(self)
```
Returns all tags in the last CSV.

Returns:
    list: display_name

### get_examples_of
```python
AutoMLHelper.get_examples_of(self, tag_name, limit=10)
```
Returns examples of tag_name in the last CSV.

Args:
    tag_name (str): tag display_name
    limit (int): max number of results

Returns:
    list: files marked with that tag

### predict_gs
```python
AutoMLHelper.predict_gs(self, gs_file)
```
Returns AutoML predictions on gs_file.

Args:
    gs_file (str): image in the gs bucket e.g.: gs://my-bucket-name/image.jpg

Returns:
    list: (`display_name`, classification score)

