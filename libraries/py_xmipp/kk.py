






print(" >>> I'm in kk.py")

def getSomething():

    return("I'm returning this")

class anyClass():

    def __init__(self):
        print("..anyClass object created.")

    @classmethod
    def someClMethod(self):
        print("I'm inside a class method!")

    def someMethod(self):
        print("I'm inside a object method!")

    def someMethod2(self):
        return  "I'm returning from object method!"