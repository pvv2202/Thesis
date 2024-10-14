class Instructions:
    '''Instruction Class'''
    def __init__(self):
        self.instructions = self.get_method_names()

    def __call__(self, stack, instruction, *args, **kwargs):
        '''Call instruction on state'''
        return getattr(self, instruction)(stack, *args, **kwargs)

    def get_method_names(self):
        '''
        Returns a set of all method names defined in the class that are not special methods.
        '''
        method_names = {
            method_name for method_name in dir(self) if callable(getattr(self, method_name)) and not method_name.startswith('__')
        }
        return method_names