from google.cloud import automl_v1beta1 as automl
from google.protobuf.json_format import MessageToJson
import json

class AutoML:

    def __init__(self, project_id, compute_region='us-central1', model_id=''):
        self.project_id = project_id
        self.compute_region = compute_region
        
        self.client = automl.AutoMlClient()
        self.project_location = self.client.location_path(
            project_id, compute_region)
        
        if model_id == '':
            model = self.get_last_model()
            self.model_id = model.name.split("/")[-1]
        else:
            self.model_id = model_id

    def get_tag_evaluation(self, model_id = ''):
        if model_id == '':
            model_id = self.model_id
        
        model_full_id = self.client.model_path(
            self.project_id, self.compute_region, model_id)
        response = self.client.list_model_evaluations(model_full_id)
        results = []
        
        for r in response:
            results.append(r)
        return results

    def get_predictions(self, image_bytes, model_id='', score_threshold= u'0.10'):
        if model_id == '':
            model_id = self.model_id

        model_full_id = self.client.model_path(
            self.project_id, self.compute_region, model_id)
        
        prediction_client = automl.PredictionServiceClient()
        
        payload = {"image": {"image_bytes": image_bytes}}
        params = {}
        if score_threshold:
            params = {"score_threshold": score_threshold}

        result = []
        predictions = prediction_client.predict(model_full_id, payload, params)
        
        for p in predictions.payload:
            result.append((p.display_name, p.classification.score))
        
        return sorted(result, key=lambda _ : _[1], reverse=True)

    def get_last_dataset(self, filter_ = ''):
        response = self.client.list_datasets(self.project_location, filter_)
        return sorted(response, key=lambda _: _.create_time.seconds, reverse=True)[0]
    
    def get_last_model(self):
        response = self.client.list_models(self.project_location)
        return sorted(response, key=lambda _: _.create_time.seconds, reverse=True)[0]

    def get_dataset_from_model(self):
        response = self.client.list_models(self.project_location)
        for model in response:
            model_name = model.name.split('/')[-1]
            if model_name == self.model_id:
                return model.dataset_id
        return None

    def get_label_name(self, dataset_id, annotation_spec_id):
        label_name = self.client.annotation_spec_path(
                    self.project_id,
                    self.compute_region,
                    dataset_id,
                    annotation_spec_id
        )
        response = self.client.get_annotation_spec(label_name)
        return response.display_name, response.example_count

    def get_labels_evaluation(self):
        import pandas as pd
        dataset_id = self.get_dataset_from_model()
        results = self.get_tag_evaluation(self.model_id)
        labels_data = {}

        for label in results:
            try:
                serialized = json.loads(MessageToJson(label))
                spec_id = serialized['annotationSpecId']
                labels_data[spec_id] = {}
                labels_data[spec_id]['auPrc'] = serialized[
                    'classificationEvaluationMetrics']['auPrc']
                labels_data[spec_id][
                    'confidenceMetricsEntry'] = serialized[
                        'classificationEvaluationMetrics']['confidenceMetricsEntry']
            except:
                continue

        results = pd.DataFrame(labels_data).T.reset_index()
        results['label'] = results['index'].apply(lambda x: self.get_label_name(dataset_id, x))
        results['best_results'] = results['confidenceMetricsEntry'].apply(self._get_best_results)
        results = self._clean_results(results)
        return results
    
    def _get_best_results(self, metrics, optimize='f1Score'):
        best_indicator = float('-inf')
        for metric in metrics:
            metric_value = float(metric.get(optimize, 0))
            if metric_value > best_indicator:
                best_indicator = metric_value

                precision = metric.get('precision', 0)
                recall = metric.get('recall', 0)
                f1 = metric.get('f1Score', 0)
                threshold = metric.get('confidenceThreshold', 0)

        return [precision, recall, f1, threshold]
    
    def _clean_results(self, df):
        df['count'] = df['label'].apply(lambda x: x[1])
        df['label'] = df['label'].apply(lambda x: x[0])
        df['precision'] = df['best_results'].apply(lambda x: x[0])
        df['recall'] = df['best_results'].apply(lambda x: x[1])
        df['f1'] = df['best_results'].apply(lambda x: x[2])
        df['threshold'] = df['best_results'].apply(lambda x: x[3])
        df.drop('best_results', axis = 1, inplace=True)
        return df