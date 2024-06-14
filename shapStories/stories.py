import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
from PIL import Image

class SHAPstory():

  def __init__(self, feature_desc, dataset_description, 
               input_description, target_description, llm=None):
    """Initializes the SHAPstory class with necessary parameters."""
    self.feature_desc = feature_desc
    self.dataset_description = dataset_description
    self.input_description = input_description
    self.target_description = target_description

    if llm is None:
      print("No language model provided. You will only be able to generate prompts.")
    else:
      self.llm = llm

  def gen_shap_feature_df(self, x, model, tree):
    
    # TODO implement for any (or at least more) model types
    if tree:
      explainer = shap.TreeExplainer(model)
      # Only takes positive shap values
      shap_vals = explainer.shap_values(x)[:, :, 1]
    else:
      # TODO update
      explainer = shap.KernelExplainer(model.predict, x)
      shap_vals = explainer.shap_values(x).T

    shap_df = pd.DataFrame(shap_vals, columns=[f"{col} SHAP Value" for col in x.columns], index=x.index)

    return pd.concat([x, shap_df], axis=1)

  def gen_variables(self,model,x,y,tree):
    """
    Generate necessary variables including SHAP values and predictions.
    
    Parameters:
    -----------
    model : object
        A trained model which supports SHAP explanations.
    x : DataFrame
        The input data.
    y : Series or array-like
        The true labels for the input data.
    tree : bool, default=True
        Boolean indicating if the model is tree-based. If True, TreeExplainer will be 
        used for SHAP explanations, else KernelExplainer will be used.
    """

    # Generate table with shap values for all instances given
    self.shap_feature_df = self.gen_shap_feature_df(x,model,tree)

    # Generate predictions and scores
    y_pred   = model.predict(x)
    y_scores = model.predict_proba(x)[:,1]
    
    self.result_df = pd.DataFrame({
      "truth" : y,
      "pred"  : y_pred,
      "score" : y_scores,
    })

  def generate_prompt(self, iloc_pos):
    """
    Generates the prompt for the provided LLM to generate a narrative.
    
    Parameters:
    -----------
    iloc_pos : int
        Index position of the instance for which the prompt is generated.
        
    Returns:
    --------
    str
        The generated prompt.
    """

    ground_truth, prediction, score = self.result_df.iloc[iloc_pos]

    feature_names = [col for col in self.shap_feature_df.columns if not col.endswith("SHAP Value")]

    row_values = self.shap_feature_df.iloc[iloc_pos]
    feature_values = row_values[[name for name in feature_names]]
    shap_values = row_values[[name + " SHAP Value" for name in feature_names]]

    row_df = pd.DataFrame({
      "Feature" : feature_names,
      "Feature Value"   : feature_values.values,
      "SHAP Value"    : shap_values.values
    })

    sorted_row_df = row_df.sort_values(by="SHAP Value", ascending=False)
  
    prompt_string = f"""
    An AI model was used to predict {self.dataset_description}. 
    The input features of the data include data about {self.input_description}. 
    The target variable is a label stating the probability that {self.target_description}.

    A certain instance in the test dataset was {'correctly classified' if prediction == ground_truth else 'misclassified'}.
    The AI model predicted a {score:.2%} probability ('{prediction}') that {self.target_description}. 
    The actual outcome was {ground_truth}. The provided SHAP table was generated to explain this
    outcome. It includes every feature along with its value for that instance, and the
    SHAP value assigned to it. 
    
    The goal of SHAP is to explain the prediction of an instance by 
    computing the contribution of each feature to the prediction. The
    SHAP explanation method computes Shapley values from coalitional game
    theory. The feature values of a data instance act as players in a coalition.
    Shapley values tell us how to fairly distribute the “payout” (= the prediction)
    among the features. A player can be an individual feature value, e.g. for tabular
    data. The scores in the table are sorted from most positive to most negative.

    Can you come up with a plausible, fluent story as to why the model could have
    predicted this outcome, based on the most influential positive and most influential
    negative SHAP values? Focus on the features with the highest absolute
    SHAP values. Try to explain the most important feature values in this story, as
    well as potential interactions that fit the story. No need to enumerate individual
    features outside of the story. Conclude with a short summary of why this
    classification may have occurred. Limit your answer to 8 sentences.

    Table containing feature values and SHAP values:
    {sorted_row_df.to_string()}

    Additional clarification of the features:
    {self.feature_desc}
    """
    
    return prompt_string

  def generate_response(self, prompt):

    return self.llm.generate_response(prompt)

  def generate_stories(self,model,x,y,tree=True):
    """
    Generates SHAPstories for each instance in the given data.
    
    Parameters:
    -----------
    model : object
        A trained model which supports SHAP explanations.
    x : DataFrame
        The input data.
    y : Series or array-like
        The true labels for the input data.
    temp : float, default=0.2
        The temperature setting for the GPT model.
    tree : bool, default=True
        Boolean indicating if the model is tree-based. If True, TreeExplainer will be 
        used for SHAP explanations, else KernelExplainer will be used.
    
    Returns:
    --------
    list of str
        A list containing the generated SHAPstories for each instance.
    """

    self.gen_variables(model, x, y, tree)

    stories = [self.generate_response(self.generate_prompt(i)) for i in range(len(x))]

    return stories
  
  def generate_stories_and_waterfall(self,model,x,y, full_x):
     
     self.gen_variables(model, x, y, tree=True)

     stories = [self.generate_response(self.generate_prompt(i)) for i in range(len(x))]
     plots = self.plot_shap_waterfall_to_array(self.shap_feature_df, model, full_x)

     return stories, plots

  # Added for the compairson between SHAPstories and waterfall plots
  def plot_shap_waterfall_to_array(self, shap_feature_df, model, full_x):
      """
      Creates SHAP waterfall plots for each instance in the DataFrame using precomputed SHAP values,
      dynamically adjusting the base (expected) value based on the predicted class,
      and stores them as image arrays. Assumes a SKLearn random forest classifier being used for
      binary classification.
      """
      image_arrays = []
      num_instances = shap_feature_df.shape[0]

      feature_names = [col for col in shap_feature_df.columns if not "SHAP Value" in col]

      predicted_probs = model.predict_proba(full_x)

      expected_value = np.mean(predicted_probs[:, 1])
      print(expected_value)

      shap_columns = [col for col in shap_feature_df.columns if "SHAP Value" in col]

      # Iterate over each data instance and create a waterfall plot
      for i in range(num_instances):
          shap_values = shap_feature_df.iloc[i][shap_columns].values

          # Create a SHAP Explanation object manually
          explanation = shap.Explanation(values=shap_values,
                                        base_values=expected_value,
                                        data=full_x.iloc[i].values,
                                        feature_names=feature_names)

          plt.figure(figsize=(10, 5))
          shap.plots.waterfall(explanation, show=False)

          # Save plot to a buffer
          buf = io.BytesIO()
          plt.savefig(buf, format="png", bbox_inches='tight')  # Use bbox_inches to prevent cut-off
          plt.close()
          buf.seek(0)
          img = Image.open(buf)
          image_array = np.array(img)
          image_arrays.append(image_array)

          buf.close()
      return image_arrays