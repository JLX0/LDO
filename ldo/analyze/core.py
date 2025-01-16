from ldo.analyze.prompts import SearchSpacePrompt
from LLM_utils.inquiry import OpenAI_interface, extract_code

class Analyzer:
    def __init__(self,api_key,model,debug):
        self.prompt_instance=SearchSpacePrompt()
        self.OpenAI_instance = OpenAI_interface(api_key, model=model, debug=debug)
    def run(self , task_description):
        response, cost= self.clarify(task_description)
        print(response)
        print(cost)
        response=extract_code(response,mode="python_object")
        response, cost = self.choose_sampler(response)
        print(response)
        print(cost)

    def clarify(self,task_description):
        self.prompt_instance.analyze_properties_prompt(task_description)
        response, cost = self.OpenAI_instance.ask(self.prompt_instance.prompt)
        return response, cost
    def choose_sampler(self,task_properties):
        self.prompt_instance.choose_sampler(task_properties)
        response, cost = self.OpenAI_instance.ask(self.prompt_instance.prompt)
        return response, cost


if __name__ == "__main__":
    chosen_test="1"

    analyzer_instance=Analyzer("","deepseek-chat",True)
    if chosen_test == "1":
        analyzer_instance.run("The dataset is the Boston Housing dataset. I am using the SVM model to predict the housing price. I want to optimize the hyperparameters for SVM for the task."
                              "I have the time to finish about 100 training and testing process of SVM")
    if chosen_test == "2":
        analyzer_instance.run("I am trying to do neural architecture search. The network being optimized is for image classification. it has about 20 layers. I can train the networks for about 200 times")

    if chosen_test == "3":
        analyzer_instance.run("I am trying to predict the stock. The network being optimized is for image classification. it has about 20 layers. I can train the networks for about 200000 times")