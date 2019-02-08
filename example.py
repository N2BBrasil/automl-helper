import os
from automlhelper import AutoMLHelper

#Initialize Object
auto = AutoMLHelper(
    bucket_name=os.environ['AUTOML_BUCKET_NAME'],
    csv_path=os.environ['AUTOML_CSV_PATH'],
    project_name=os.environ['AUTML_PROJECT_NAME'])

#Return 10 best performing tags
best_performing = auto.best_performing(limit=10, reverse=True)
print(best_performing)

#Return 10 worst performing tags
worst_performing = auto.best_performing(limit=10, reverse=False)
print(worst_performing)

#Get prediction results
gs_file = auto.get_examples_of(best_performing[0][0])[0]
print(gs_file)
print(auto.predict_gs(link))