import sympy
# from sympy import symbols, diag, zeros

def get_perm(a, b):
    perm = []
    for k in a:
        perm.append(b.index(k))
    return perm

def symb_sign(s):
    if type(s) == sympy.core.symbol.Symbol:
        return 1
    else:
        return -1

class Index:
    def __init__(self, s):
        self.s = s

    def sign(self):
        if type(self.s) == sympy.core.symbol.Symbol:
            return 1
        else:
            return -1

    def move(self):
        if self.sign() == 1:
            return Index(-self.s)
        else:
            return Index(self.s.args[1])

    def __eq__(self, other):
        return self.s == other.s

    def __abs__(self):
        if self.sign() == 1:
            return self.s
        else:
            return self.s.args[1]

    def __hash__(self):
        return hash(self.s)

    def __repr__(self):
        return repr(self.s)

class Tensor:
    def __init__(self, array, symbols):
        self.array = array
        self.symbols = symbols
        self.indices = [Index(s) for s in symbols]
        self.abs_indices = [abs(ind) for ind in self.indices]
        self.upper = [abs(ind) for ind in self.indices if ind.sign() == 1]
        self.lower = [abs(ind) for ind in self.indices if ind.sign() == -1]
        self.metric = sympy.Array(sympy.diag(1, -1, -1, -1))

    def _repr_latex_(self):
        if len(self.array.shape) == 1:
            if self.indices[0].sign() == 1:
                return sympy.Matrix(self.array).reshape(self.array.shape[0], 1)._repr_latex_()
            if self.indices[0].sign() == -1:
                return sympy.Matrix(self.array).reshape(1, self.array.shape[0])._repr_latex_()
        if len(self.array.shape) == 2:
            return sympy.Matrix(self.array)._repr_latex_()

        return self.array._repr_latex_()

    def perm(self, *symbols):
        indices = [Index(s) for s in symbols]
        if (set(self.indices) == set(indices)):
            perm = get_perm(self.indices, indices)
            return Tensor(sympy.permutedims(self.array, perm), symbols)
        raise ValueError("indices must be the same")

    def __add__(self, other):
        t = other.perm(*self.symbols)
        return Tensor(self.array + t.array, self.symbols)

    def __sub__(self, other):
        t = other.perm(*self.symbols)
        return Tensor(self.array - t.array, self.symbols)

    def mul_contract(self, other):
        mv_symbols = [ind.move().s for ind in other.indices]
        intersection = set(self.symbols) & set(mv_symbols)
        contractions = []
        new_symbols = []
        n = len(self.symbols)
        for i, a in enumerate(self.symbols):
            if a in intersection:
                k = mv_symbols.index(a)
                contractions.append((i, n + k))
            else:
                new_symbols.append(a)

        for a in other.indices:
            if a.move().s not in intersection:
                new_symbols.append(a.s)

        return Tensor(sympy.tensorcontraction(sympy.tensorproduct(self.array, other.array), *contractions), new_symbols)


    def move(self, i):
        new_symbols = list(self.symbols)
        n = len(self.symbols)
        new_symbols[i] = self.indices[i].move().s
        return Tensor(sympy.tensorcontraction(sympy.tensorproduct(self.array, self.metric), (i, n)), new_symbols)


    def __mul__(self, other):
        return self.mul_contract(other)
        # if len(set(self.abs_indices) & set(other.abs_indices)) == 0:
        #     array = sympy.tensorproduct(self.array, other.array)
        #     symbols = [ind.s for ind in (self.indices + other.indices)]
        #     return Tensor(array, symbols)

    def __rmul__(self, other):
        if type(other) is not Tensor:
            return Tensor(other * self.array, self.symbols)

    def __getitem__(self, args):
        if type(args) is not tuple:
            args = [args]
        nums = []
        num_symbols = []
        symbols = []
        arg_symbols = []
        T = self
        for i, a in enumerate(args):
            if type(a) is int:
                nums.append(a)
                num_symbols.append(self.symbols[i])
            else:

                arg_symbols.append(a)
                if symb_sign(a) != symb_sign(self.symbols[i]):
                    T = T.move(i)
                    symbols.append(self.indices[i].move().s)
                else:
                    symbols.append(self.symbols[i])

        # print('nums', nums)
        # print('num_symbols', num_symbols)
        # print('symbols', symbols)
        # print('arg_symbols', arg_symbols)

        t = T.perm(*(num_symbols + symbols))
        return Tensor(t.array[tuple(nums)], arg_symbols)




