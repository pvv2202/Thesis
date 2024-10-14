from instructions import Instructions


class FloatInstructions(Instructions):
    '''Float Instruction Class'''

    def __init__(self):
        super().__init__()

    def __call__(self, stack, instruction):
        if hasattr(self, instruction):
            return getattr(self, instruction)(stack)
        else:
            raise AttributeError(f"FloatInstruction does not have {instruction} method")

    '''
    Float Operations
    '''
    @staticmethod
    def float_add(stack):
        '''Adds two floats'''
        if len(stack['int']) >= 2:
            a = stack['int'].pop()
            b = stack['int'].pop()
            stack['int'].append(a + b)

    @staticmethod
    def float_mult(stack):
        '''Multiplies two floats'''
        if len(stack['int']) >= 2:
            a = stack['int'].pop()
            b = stack['int'].pop()
            stack['int'].append(a * b)

    @staticmethod
    def float_divide(stack):
        # TODO: Might need to be careful here so we don't get enormous values dividing by something
        '''Divides two floats'''
        if len(stack['int']) >= 2:
            # Check so we don't divide by zero
            if stack['int'][-1] != 0:
                a = stack['int'].pop()
                b = stack['int'].pop()
                stack['int'].append(a // b)

    @staticmethod
    def float_sqrt(stack):
        '''Takes the square root of a floats'''
        if len(stack['int']) >= 1:
            a = stack['int'].pop()
            stack['int'].append(a ** 0.5)