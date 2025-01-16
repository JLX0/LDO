from LLM_utils.prompter import PromptBase
from ldo.load.load_optuna import get_sampler_names


class SearchSpacePrompt(PromptBase) :
    BBO_task_properties = {
        "smoothness" : "Whether the objective function is smooth or non-smooth (e.g., piecewise or discontinuous)." ,
        "noise in evaluation" : "Whether the objective function evaluations are noisy or stochastic, which can affect the reliability of optimization." ,
        "budget" : "The computational budget available, such as the maximum number of function evaluations or time constraints." ,
        "constraint type" : "The nature of constraints in the problem, such as linear, nonlinear, equality, or inequality constraints." ,
        "continuous vs discrete vs mixed" : "Whether the problem involves continuous variables, discrete variables, or a mix of both." ,
        "single objective vs multi-objective" : "Whether the task involves optimizing a single objective function or multiple (possibly conflicting) objectives." ,
        "modality" : "Whether the objective function is unimodal (having a single optimum) or multimodal (having multiple local optima). Multimodal problems require algorithms that can escape local optima."
        }

    def __init__(self) :
        self.samplers = get_sampler_names()

    def analyze_properties_prompt(self , user_description) :
        prompt_string = [
            "Given a description of an optimization task, your goal is to determine which optimization algorithm to use." ,
            "You need to consider the following aspects of the optimization problem:"
            ]

        # Add the BBO task properties to the prompt
        for key , description in self.BBO_task_properties.items() :
            prompt_string.append(f"- {key}: {description}")

        prompt_string.append(
            "You will need to decide which algorithm to use among the following list of algorithms:")

        prompt_string.append("Here is the description of the optimization task from the user:")

        prompt_string.append(user_description)

        prompt_string.append(
            f"Your answer should be either a Python dictionary. The dictionary includes and only includes the following keys: {self.BBO_task_properties.keys()}. Each value is "
            f"an analysis of the corresponding properties.")


        prompt_string.append("Your answer should not include any additional information, such as"
                             "introduction, explanation, or context")
        prompt_string.append("Here is the Python dictionary:")

        self.prompt=self.list_to_formatted_OpenAI(prompt_string)

    def choose_sampler(self , task_properties) :
        prompt_string = [
            "Given the properties of the optimization task, your goal is to choose the most suitable sampler from the available list." ,
            "The properties of the optimization task are as follows:"
            ]

        # Add the task properties to the prompt
        for key , value in task_properties.items() :
            prompt_string.append(f"- {key}: {value}")

        prompt_string.append("The available samplers are:")

        # Add the list of available samplers to the prompt
        for sampler in self.samplers :
            prompt_string.append(f"- {sampler}")

        prompt_string.append(
            "Your task is to choose the most suitable sampler based on the properties of the optimization task. "
            "Your answer should be a single string containing the name of the chosen sampler."
            )

        prompt_string.append("Here is the chosen sampler:")

        self.prompt = self.list_to_formatted_OpenAI(prompt_string)
