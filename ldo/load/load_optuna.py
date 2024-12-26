import optuna
import inspect

def get_sampler_names():
    """
    Retrieves the list of sampler class names available in optuna.samplers.

    Returns:
        list: A list of strings, each representing a sampler class name.
    """
    sampler_classes = inspect.getmembers(optuna.samplers, inspect.isclass)
    samplers = [cls_name for cls_name, cls_obj in sampler_classes if "Sampler" in cls_name]
    return samplers

def get_sampler_details(sampler_name):
    """
    Retrieves the documentation and source code for a specified sampler.

    Args:
        sampler_name (str): The name of the sampler (e.g., "BaseSampler").

    Returns:
        dict: A dictionary with keys "documentation" and "source_code", containing
              the respective information for the specified sampler.
    """
    sampler_classes = inspect.getmembers(optuna.samplers, inspect.isclass)
    samplers_dict = {cls_name: cls_obj for cls_name, cls_obj in sampler_classes}

    if sampler_name in samplers_dict:
        sampler_class = samplers_dict[sampler_name]
        documentation = inspect.getdoc(sampler_class) or "No documentation available."
        try:
            source_code = inspect.getsource(sampler_class)
        except OSError:
            source_code = "Source code not available."
        return {"documentation": documentation, "source_code": source_code}
    else:
        return {"error": f"Sampler '{sampler_name}' not found."}

if __name__ == "__main__":

    sampler_names = get_sampler_names()
    print(sampler_names)
    # Output: ['BaseSampler', 'RandomSampler', 'TPESampler', ...]

    sampler_details = get_sampler_details("TPESampler")
    print(sampler_details["documentation"])  # Print the docstring
    print(sampler_details["source_code"])    # Print the source code
