from automlhelper.storageio import StorageIO
from automlhelper.automl import AutoML

class AutoMLHelper:
    """AutoML Helper

    Args:
        bucket_name (int): The name of the bucket utilized in AutoML e.g.: my-bucket-vcm
        csv_path (str): The path where yours CSVs with reference of images and classes of your images. e.g.: csv/
        project_name (str): Your Google Cloud project 
    """

    def __init__(self, bucket_name, csv_path, project_name):
        self.bucket_name = bucket_name
        self.csv_path = csv_path
        self.project_name = project_name
        self.automl = AutoML(project_name)
        self.storageio = StorageIO(bucket_name, csv_path)
        self.performance = False
        self.tags = False

    def tag_performance(self):
        """Returns tag `display_name` and performance object.

        Warning: This connection between tag display_name and the performance object is not returned by Google`s API, this method utilizes an alternative way that can stop working at any moment.

        Returns:
            list: (display_name, performance_object)
        """

        if self.performance == False:
            self.performance = self.automl.get_tag_evaluation()

        if self.tags == False:
            self.tags = sorted(self.get_tags(),reverse=True)
        
        return zip(self.tags, self.performance)
    
    def tag_and_performance(self, key='base_au_prc'):
        """Returns tag `display_name` and performance metric.

        Args:
            key (str): performance metric of performance object defaults: 'base_au_prc'

        Returns:
            list: (display_name, performance_metric)
        """

        values = []
        for t,p in self.tag_performance():
            values.append((t, getattr(p.classification_evaluation_metrics, key)))
        return values

    def best_performing(self, key='base_au_prc', limit=0, reverse=True): 
        """Returns best or worst tags by performance metric.

        Args:
            key (str): performance metric of the performance object defaults: 'base_au_prc'
            limit (int): max number of results
            reverse (bool): True for best performing, False for worst performing

        Returns:
            list: (display_name, performance_metric)
        """
        
        values = sorted(self.tag_and_performance(key), key=lambda _: _[1], reverse=reverse)
        
        if limit > 0:
            return values[0:limit]
        else:
            return values

    def get_tags(self):
        """Returns all tags in the last CSV.

        Returns:
            list: display_name
        """

        lines = self.storageio.get_csv_lines()
        tags = []
        for l in lines:
            for t in l.split(','):
                if 'gs://' not in t and t not in tags + [
                    'TRAIN','TEST','VALIDATION']:
                    tags.append(t)

        return sorted(tags)

    def get_examples_of(self, tag_name, limit=10):
        """Returns examples of tag_name in the last CSV.

        Args:
            tag_name (str): tag display_name
            limit (int): max number of results

        Returns:
            list: files marked with that tag
        """

        csv_lines = self.storageio.get_csv_lines()
        files = []

        for line in csv_lines:
            if len(files) > limit:
                break
            
            breaks = line.split(',')

            if tag_name in breaks:
                gs_file = list(filter(lambda _: 'gs://' in _,breaks))[0]
                files.append(self.storageio.gs_to_link(gs_file))
        return files

    def predict_gs(self, gs_file):
        """Returns AutoML predictions on gs_file.

        Args:
            gs_file (str): image in the gs bucket e.g.: gs://my-bucket-name/image.jpg

        Returns:
            list: (`display_name`, classification score)
        """       

        return self.automl.get_predictions(
            self.storageio.get_base64_from_gs(gs_file)
        )