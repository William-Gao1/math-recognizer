import random
class QuestionGenerator:
    
    def __init__(self):
        self.operations = [self.multiply, self.add, self.subtract]

    def get_operation(self):
        return random.choice(self.operations)

    def multiply(self, a, b):
        return a * b

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
    
    def create_problem (self):
        operation = self.get_operation()
        if operation == self.multiply:
            first_num = random.randint(0,11)
            second_num = random.randint(0,11)
            question = str(first_num)+ " x " + str(second_num)
        elif operation == self.add:
            first_num = random.randint(0,50)
            second_num = random.randint(0,51)
            question = str(first_num)+ " + " + str(second_num)
        else:
            first_num = random.randint(0,51)
            second_num = random.randint(0,50)
            first_num , second_num = (second_num, first_num) if first_num < second_num else (first_num, second_num)
            question = str(first_num)+ " - " + str(second_num)
        answer = operation(first_num, second_num)

        return question, answer

gen = QuestionGenerator()
print(gen.create_problem())
