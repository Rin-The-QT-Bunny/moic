class Distribution:
    def __init__(self):
        pass
    def __eq__(self,o):return True

    def __ne__(self,o): return not o == self

    def __str__(self): return "Distribution"