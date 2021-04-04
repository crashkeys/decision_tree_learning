from colorama import Fore
from colorama import Style

class DecisionLeaf:

    def __init__(self, output):
        self.output = output

    def classify(self):
        return self.output

    def display(self):
        print(f"{Fore.LIGHTGREEN_EX} No uncertainty! OUTPUT = {self.output}{Style.RESET_ALL}")
        print(" ")


