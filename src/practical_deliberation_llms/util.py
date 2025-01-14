import numpy as np
import copy

from langchain_core.messages import AIMessage


def get_labelprobs_from_message(result_obj: AIMessage, labels=["a", "b"]):    
    """
    Extracts label probabilities from the response of a language model.

        Parameters
        ----------
        result_obj : AIMessage
            The AIMessage object containing the response metadata with log probabilities.
        labels : list of str, optional
            A list of label strings to match against the tokens in the response. Default is ["a", "b"].

        Returns
        -------
        dict
            A dictionary where keys are labels and values are the summed probabilities of the tokens matching each label.
    """
    first_logprobs = copy.copy(result_obj.response_metadata["logprobs"]["content"][0]["top_logprobs"])
    for record in first_logprobs:
        matches = [l for l in labels if l in record["token"].lower()]
        if len(matches) == 1:
            record["label"] = matches[0]
        else:
            record["label"] = None

    # logprobs to probs via softmax
    logprobs = [record["logprob"] for record in first_logprobs]
    probs = np.exp(logprobs) / np.exp(logprobs).sum()
    for record, prob in zip(first_logprobs, probs):
        record["prob"] = prob
    label_probs = {
        label: sum([record["prob"] for record in first_logprobs if record["label"] == label])
        for label in labels
    }
    return label_probs
