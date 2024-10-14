from instructions import Instructions

class IntInstructions(Instructions):
    '''Int Instruction Class'''
    def __init__(self):
        super().__init__()

    def __call__(self, stack, instruction):
        if hasattr(self, instruction):
            return getattr(self, instruction)(stack)
        else:
            raise AttributeError(f"IntInstruction does not have {instruction} method")

    '''
    Int Operations
    '''
    @staticmethod
    def int_add(stack):
        '''Adds two integers'''
        if len(stack['int']) >= 2:
            a = stack['int'].pop()
            b = stack['int'].pop()
            stack['int'].append(a + b)
            
    @staticmethod
    def int_mult(stack):
        '''Multiplies two integers'''
        if len(stack['int']) >= 2:
            a = stack['int'].pop()
            b = stack['int'].pop()
            stack['int'].append(a * b)

    @staticmethod
    def int_divide(stack):
        '''Divides two integers'''
        if len(stack['int']) >= 2:
            a = stack['int'].pop()
            b = stack['int'].pop()
            stack['int'].append(a // b)

    @staticmethod
    def int_sqrt(stack):
        '''Takes the square root of an integer'''
        if len(stack['int']) >= 1:
            a = stack['int'].pop()
            stack['int'].append(a ** 0.5)

