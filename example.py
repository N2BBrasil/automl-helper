import os
from automlhelper import AutoMLHelper

auto = AutoMLHelper(
    bucket_name=os.environ['AUTOML_BUCKET_NAME'],
    csv_path=os.environ['AUTOML_CSV_PATH'],
    project_name=os.environ['AUTML_PROJECT_NAME'])

best_performing = auto.best_performing(limit=10, reverse=False)

print(best_performing)
print(auto.best_performing(limit=10, reverse=True))

link = auto.get_examples_of(best_performing[0][0])[0]
print(link)
print(auto.predict_gs(link))