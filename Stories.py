import pandas as pd
import shap
import openai

class SHAPstory():
  """
    A class to generate SHAPstories, narratives that explain AI predictions based on SHAP values.
    
    Attributes:
    -----------
    api_key : str
        The OpenAI API key for generating narratives.
    feature_desc_df : DataFrame
        A DataFrame containing descriptions for each feature.
    dataset_description : str
        A brief description of the dataset.
    input_description : str
        Description of the input features.
    target_description : str
        Description of the target variable.
    gpt_model : str, default="gpt-4"
        The model to be used from OpenAI.
    """

  def __init__(self, api_key, feature_desc_df, dataset_description, 
               input_description, target_description, gpt_model = "gpt-4"):
    """Initializes the SHAPstory class with necessary parameters."""
    self.api_key = api_key
    self.feature_desc_df = feature_desc_df
    self.dataset_description = dataset_description
    self.input_description = input_description
    self.target_description = target_description
    self.gpt_model = gpt_model

  def gen_shap_feature_df(self, x, model, tree):
    """
    Generates a DataFrame with original features and their corresponding SHAP values.
    
    Parameters:
    -----------
    x : DataFrame
        The input data for which SHAP values are to be calculated.
    model : object
        A trained model which supports SHAP explanations.
    tree : bool, default=True
        Boolean indicating if the model is tree-based. If True, TreeExplainer will be 
        used for SHAP explanations, else KernelExplainer will be used.
    
    Returns:
    --------
    DataFrame
        A DataFrame with original features and their SHAP values.
    """

    if tree:
      explainer = shap.TreeExplainer(model)
      shap_vals = explainer.shap_values(x)[1].T
    else:
      explainer = shap.KernelExplainer(model.predict, x)
      shap_vals = explainer.shap_values(x).T

    df_copy = x.copy()

    for i,col in enumerate(df_copy.columns):
      df_copy[f"{col} SHAP Value"] = shap_vals[i]

    return df_copy

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
      "score" : y_scores
    })

  def generate_prompt(self, iloc_pos):
    """
    Generates the prompt for OpenAI's GPT model to generate a narrative.
    
    Parameters:
    -----------
    iloc_pos : int
        Index position of the instance for which the prompt is generated.
        
    Returns:
    --------
    str
        The generated prompt.
    """

    prediction, ground_truth, score = self.result_df.iloc[iloc_pos]
  
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

    SHAP table:
    {self.shap_feature_df.iloc[iloc_pos].to_string()}

    Additional clarification of the features:
    {self.feature_desc_df.to_string(index = False)}
    """

    return prompt_string

  def generate_response(self, prompt, temp):
    """
    Gets the response from the OpenAI's GPT model based on a given prompt.
    
    Parameters:
    -----------
    prompt : str
        The generated prompt for which the narrative is to be generated.
    temp : float, default=0.2
        The temperature setting for the GPT model.
        
    Returns:
    --------
    Response object
        The response from OpenAI's API.
    """

    openai.api_key = self.api_key

    message=[{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=self.gpt_model,
        messages = message,
        temperature=temp,
        max_tokens = 1000
    )

    return response.choices[0]["message"]["content"]

  def generate_stories(self,model,x,y,temp=0.2,tree=True):
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

    stories = [self.generate_response(self.generate_prompt(i), temp) for i in range(len(x))]

    return stories