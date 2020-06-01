import sys
import copy
import parser
import random
from typing import List, Optional, Union
class Cell:
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.type = type

class Robot:
	def __init__(self, file, drons = 10000):
		self.life = True
		s = file.readline()
		b=[]
		self.x=int(s[0])
		self.y=int(s[2])
		c=file.read()
		b=c.split('\n')
		for i in range(len(b)):
			b[i]=[int(j) for j in b[i]]
		self.map =b
		self.drons = drons
        
	def show_map(self):
		for i in range(len(self.map)):
			for j in range(len(self.map[0])):
				if i == self.x and j == self.y:
					print("0", end='') #Robot
				elif self.map[i][j] == 1: #Wall
					print("#", end='')	
				elif self.map[i][j]  ==3: #Exit
					print("X", end='')
				else:          #Empty
					print(" ", end='')			
			print()
			
	def up(self, n):
		for i in range(n):
			if self.life:
				self.x-=1
				if self.x<0 or self.x>=len(self.map):
					self.life=False
				elif self.map[self.x][self.y] == 1:
					self.life=False
		self.show_map()

	def down(self, n):
		for i in range(n):
			if self.life:
				self.x+=1
				if self.x<0 or self.x>=len(self.map):
					self.life=False
				elif self.map[self.x][self.y] == 1:
					self.life=False
		self.show_map()

	def left(self, n):
		for i in range(n):
			if self.life:
				self.y-=1
				if self.y<0 or self.y>=len(self.map[0]):
					self.life=False
				if self.map[self.x][self.y] == 1:
					self.life=False
		self.show_map()

	def right(self, n):
		for i in range(n):
			if self.life:
				self.y+=1
				if self.y<0 or self.y>=len(self.map[0]):
					self.life=False
				if self.map[self.x][self.y] == 1:
					self.life=False
		self.show_map()

	def drons_count(self):
		return self.drons
    
	def send_drons(self, n):
		self.drons-=n
		new_map=[]
		array=[0]*121
		for i in range(n):
			a=Satellite(self.x, self.y, self.map)
			new_map.append(a.exploring())    
		for row in new_map:
			for cell in row:
				a=cell.x-self.x+5
				b=cell.y-self.y+5
				array[a*11+b]=cell.type
		return array
        
        

    
class Satellite:
	def __init__(self, x, y, _map):
		self.life = True
		self.x = x
		self.y = y
		self.map = _map
		self.new_map = []
	def up(self):
		self.x-=1
		if (self.x<0 or self.x>=len(self.map)):
			self.life=False
			self.new_map.append(Cell(self.x,self.y,1))
		else:
			if self.map[self.x][self.y] == 1:
				self.life=False
			self.new_map.append(Cell(self.x,self.y,self.map[self.x][self.y]))
		
	def down(self):
		self.x+=1
		if (self.x<0 or self.x>=len(self.map)):
			self.life=False
			self.new_map.append(Cell(self.x,self.y,1))
		else:
			if self.map[self.x][self.y] == 1:
				self.life=False
			self.new_map.append(Cell(self.x,self.y,self.map[self.x][self.y]))
		
	def left(self):
		self.y-=1
		if (self.y<0 or self.y>=len(self.map[0])):
			self.life=False
			self.new_map.append(Cell(self.x,self.y,1))
		else:
			if self.map[self.x][self.y] == 1:
				self.life=False
			self.new_map.append(Cell(self.x,self.y,self.map[self.x][self.y]))
		
	def right(self):
		self.y+=1
		if (self.y<0 or self.y>=len(self.map[0])):
			self.life=False
			self.new_map.append(Cell(self.x,self.y,1))
		else:
			if self.map[self.x][self.y] == 1:
				self.life=False
			self.new_map.append(Cell(self.x,self.y,self.map[self.x][self.y]))
		
	def exploring(self):
		steps=random.randint(1,5)
		for i in range(steps):
			if self.life:
				step=random.randint(1,4)
				if step == 1:
					self.up()
				elif step == 2:
					self.down()
				elif step == 3:
					self.left()
				else:
					self.right()
			else:
				break
		return self.new_map

class Variable:
	
	def __init__(self, symtype='var', value=None, const_flag=False, dim=0,dims=[]):
		self.type = symtype
		self.value = value
		self.const_flag = const_flag
		self.dim=dim
		self.dims=dims
        
	def __repr__(self):  
		if self.dim != 0:
			values = []
			for var in self.value:
				values.append(var.value)
			return f'{self.type}, {values}, {self.const_flag}, {self.dim}, {self.dims}'
		else:
			if self.type == 'BOOL':
				if self.value is True:
					self.value = 'TRUE'
				else:
					self.value = 'FALSE'
			return f'{self.type}, {self.value}, {self.const_flag}'

	def __deepcopy__(self, memodict={}):
		return Variable(self.type, self.value, self.const_flag, self.dim, self.dims)
	
	def __lt__(self, other):
		if self.value < other.value:
			return True
		return False

	def __gt__(self, other):
		if self.value > other.value:
			return True
		return False

	def __eq__(self, other):
		if self.value == other.value:
			return True
		return False

	def __bool__(self):
		return bool(self.value)
		
	def index_recount(self, indexes: list):
		n=0
		s=len(self.value)
		for i in range(len(indexes)):
			s/=self.dims[i]
			n+=indexes[i]*s
		return int(n)
		
class Interpreter:

	def __init__(self,file):
		self.parser = parser.Parser()
		self.program = None
		self.symbol_table = [dict()]
		self.functions = dict()
		self.tree = None
		self.scope = 0
		self.robot=Robot(file)

	def interpreter(self, prog=None):
		self.prog = prog
		self.tree, self.functions, parsing_ok = self.parser.parse(self.prog)
		if parsing_ok:
			self.interpreter_node(self.tree)
			if 'main' not in self.functions.keys():
				sys.stderr.write(f'error: no main function\n ')
				return
			else:
				self.scope=1
				self.symbol_table.append(dict())
				self.interpreter_node(self.functions['main'])
				
		else:
			sys.stderr.write(f'Can\'t intemperate this, incorrect syntax\n')

	def interpreter_tree(self, tree):
		print("Program tree:\n")
		tree.print()
		print("\n")

	def interpreter_node(self, node: parser.SyntaxTreeNode):
		if node is None:
			return
		if node.type == 'program':
			self.interpreter_node(node.children)
		elif node.type == 'blocks' or node.type == 'conditions' or node.type == 'statements' or node.type == 'declarations':
			for child in node.children: 
				self.interpreter_node(child)
		elif node.type == 'vardeclaration' or node.type == 'switch':
			self.interpreter_node(node.children)
		elif node.type == 'declaration_var':
			name = node.children.value
			if (name in self.symbol_table[self.scope].keys()) or (name in self.symbol_table[0].keys()):
				sys.stderr.write(f'error: redeclaration of variable {name} in {node.lineno}\n')
				return
			else:
				self.symbol_table[self.scope][name] = Variable(node.value.value, None, False)
		elif node.type == 'declaration_var_init':
			self.initialization(node, False)
		elif node.type == 'declaration_var_const':
			self.initialization(node, True)
		elif node.type == 'declaration_array':
			self.array_declaration(node, False)
		elif node.type == 'declaration_array_init': 
			self.array_initialization(node, False)
		elif node.type == 'declaration_array_const':
			self.array_initialization(node, True)
		elif node.type == 'function':
			if node.value.value not in self.functions.keys():
				self.functions[node.value.value] = node.children 
			else:                                                        
				sys.stderr.write(f'error: redeclaration of function {node.value.value} in {node.lineno}\n')
		elif node.type == 'function_call': 
			self.function_call(node)
		elif node.type == 'assignment':
			self.assignment(node)
		elif node.type == 'variables' or node.type == 'dimensions' or node.type == 'indexes' or node.type == 'values': 
			variables = []
			for child in node.children:
				var = self.interpreter_node(child)
				if var is not None:
					variables.extend(var)
				else: return None
			return variables
		elif node.type == 'variable':
			name = node.value.value
			return self.find_variable(name)
		elif node.type == 'variable_array':
			name = node.value.value
			indexes = self.interpreter_node(node.children)
			return self.find_elem_of_array(name, indexes)
		elif node.type == 'const':
			value = node.value
			if isinstance(value, int):
				return Variable('INT', value, True)
			elif value == 'FALSE':
				return Variable('BOOL', False, True)
			elif value == 'TRUE':
				return Variable('BOOL', True, True)
			else:
				return Variable('CELL', value, True) #EMPTY/WALL/EXIT/UNDEF
		elif node.type == 'expressions':			
			expressions = [] 
			for child in node.children:
				expr = self.interpreter_node(child)
				if expr is not None:
					if isinstance(expr,list):
						expressions.extend(expr)
					else:
						expressions.append(expr)
				else: return None
			return expressions 
		elif node.type == 'dimension':
			expr = self.interpreter_node(node.children)
			if isinstance(expr,list):
				expr = expr[0]
			if expr.dim != 0:
				sys.stderr.write(f'error: dimension count cannot be initialized with array in {node.lineno}\n')
				return
			if expr.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to INT to get the dimensions count in {node.lineno}\n')
				return
			return [int(expr.value)]
		elif node.type == 'index':
			expr = self.interpreter_node(node.children) 
			if isinstance(expr,list):
				expr = expr[0]
			if expr.dim != 0:
				sys.stderr.write(f'error: index cannot be initialized with array in {node.lineno}\n')
				return
			if expr.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to INT to get the index in {node.lineno}\n')
				return
			return [int(expr.value)]
		elif node.type == 'value':
			expr = self.interpreter_node(node.children)
			if isinstance(expr,list):
				expr = expr[0]
			if expr.dim != 0:
				sys.stderr.write(f'error: value cannot be initialized with array in {node.lineno}\n')
				return
			return [expr]	
		elif node.type == 'condition':
			condition = self.interpreter_node(node.children['condition'])
			if isinstance(condition,list):
				condition = condition[0]
			if condition.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to BOOL to check a condition in {node.lineno}\n')
				return
			if condition.dim != 0:
				sys.stderr.write(f'error: array cannot be condition in {node.lineno}\n')
				return
			if bool(condition.value) is True:
				self.interpreter_node(node.children['body'])
		elif node.type == 'while':
			while True:
				condition = self.interpreter_node(node.children['condition'])
				if isinstance(condition,list):
					condition = condition[0]
				if condition.type == 'CELL':
					sys.stderr.write(f'error: cannot convert CELL to BOOL to check a condition in {node.lineno}\n')
					return
				if condition.dim != 0:
					sys.stderr.write(f'error: array cannot be condition in {node.lineno}\n')
					return
				if bool(condition.value) is True:
					self.interpreter_node(node.children['body'])
				else:
					break
		elif node.type == 'math': #проверять expression на немассивность
			if node.value == '<ADD>':
				return self.addition(node.children)
			elif node.value == '<MUL>':
				return self.multiplication(node.children)
			elif node.value == '<SUB>':
				return self.subtraction(node.children)
			elif node.value == '<DIV>':
				return self.division(node.children)
			elif node.value == '<MAX>':
				return self.maximum(node.children)
			elif node.value == '<MIN>':
				return self.minimum(node.children)
			elif node.value == '<EQ>':
				return self.equality(node.children)
			elif node.value == '<OR>':
				return self.logic_or(node.children)
			elif node.value == '<AND>':
				return self.logic_and(node.children)
			elif node.value == '<NOT>':
				return self.logic_not(node.children)
		elif node.type == 'operator':
			if node.value == '<LEFT>':
				n=self.interpreter_node(node.children)
				self.robot.left(n.value)
			elif node.value == '<RIGHT>':
				n=self.interpreter_node(node.children)
				self.robot.right(n.value)
			elif node.value == '<UP>':
				n=self.interpreter_node(node.children)
				self.robot.up(n.value)
			elif node.value == '<DOWN>':
				n=self.interpreter_node(node.children)
				self.robot.down(n.value)
			elif node.value == '<GETDRONSCOUNT>':
				var = self.interpreter_node(node.children)
				if isinstance(var,list):
					var = var[0]
				n=self.robot.drons_count()
				if var.dim !=0 or var.type != 'INT':
					sys.stderr.write(f'error: assignment error in {node.lineno}\n')
					return
				var.value=n
		elif node.type == 'senddrons':
			n=self.interpreter_node(node.children)
			array=self.robot.send_drons(n.value)
			cells=[]
			a=['UNDEF','WALL','EMPTY','EXIT']
			for i in array:
				cells.append(Variable('CELL', a[i], False))
			return Variable('CELL', cells, True, 2, [11,11])

	def assignment(self, node: parser.SyntaxTreeNode): 
		variables = self.interpreter_node(node.children[1])
		if variables is None:
			sys.stderr.write(f'error: incorrect variables in assignment in {node.lineno}\n')
			return
		expression = self.interpreter_node(node.children[0])
		if expression is None:
			sys.stderr.write(f'error: incorrect expression in assignment in {node.lineno}\n')
			return 
		if isinstance(expression,list):
			expression = expression[0]
		arr = 0
		for var in variables:
			if var.dim != 0:
				arr = arr + 1
		if expression.dim == 0 and arr == 0: #везде немассивы
			for var in variables:
				if var.const_flag is True:
					sys.stderr.write(f'error: cannot reinitialize the constant in {node.lineno}\n')
					return
				conversed_var = self.type_conversion(var, expression)
				if conversed_var is None: 
					sys.stderr.write(f'error: assignment error in {node.lineno}\n')
					return
				var.type = conversed_var.type
				var.value = conversed_var.value
		elif expression.dim != 0 and arr == len(variables) and arr != 0: #везде массивы
			for var in variables:
				if var.const_flag is True:
					sys.stderr.write(f'error: cannot reinitialize the constant in {node.lineno}\n')
					return
				if var.dims != expression.dims:
					sys.stderr.write(f'error: different sizes of arrays in assignment in {node.lineno}\n')
					return
				var.value = copy.deepcopy(expression.value)
		else:
			sys.stderr.write(f'error: cannot assign array to not array and vice versa(i naoborot) in {node.lineno}\n')
			return
	
	def find_variable(self, name):
		if name in self.symbol_table[0].keys():
			return [self.symbol_table[0][name]]
		if name in self.symbol_table[self.scope].keys():
			return [self.symbol_table[self.scope][name]]
		sys.stderr.write(f'error: undeclarated variable {name}\n')
		
	def find_elem_of_array(self, name, indexes: list):
		var = self.find_variable(name)
		if var[0].dim == 0: #немассив
			sys.stderr.write(f'error: variable {name} is not an array\n')
			return
		i = var[0].index_recount(indexes)
		if var[0].const_flag is True:
			var[0].value[i].const_flag=True
		return [var[0].value[i]]
	
	def bool_to_int(self, var: Variable):
		var.type = 'INT'
		if var.value is True:
			var.value = 1
		else:
			var.value = 0
		return var
		
	def int_to_bool(self, var: Variable):
		var.type = 'BOOL'
		if var.value == 0:
			var.value = False
		else:
			var.value = True
		return var
		
	def type_conversion(self, var1: Variable, var2: Variable):
		var2.const_flag = var1.const_flag
		if var1.type != var2.type:
			if (var1.type == 'CELL') or (var2.type == 'CELL'):
				sys.stderr.write(f'error: cannot convert {var2.type} to {var1.type}\n')
				return
			elif var2.type == 'BOOL':
				return self.bool_to_int(var2)
			elif var2.type == 'INT':
				return self.int_to_bool(var2)
		else:
			return var2
		
	def initialization(self, node: parser.SyntaxTreeNode, flag: bool):
		vartype = node.value.value
		name = node.children[0].value
		if (name in self.symbol_table[self.scope].keys()) or (name in self.symbol_table[0].keys()):
			sys.stderr.write(f'error: redeclaration of variable {name} in {node.lineno}\n')
			return
		expr = self.interpreter_node(node.children[1])
		if isinstance(expr,list):
			expr = expr[0]	
		var = self.type_conversion(Variable(vartype, None, flag), expr)
		if var is None: 
			sys.stderr.write(f'error: initialization error of variable {name} in {node.lineno}\n')
			return 
		self.symbol_table[self.scope][name] = var

	def array_declaration(self, node: parser.SyntaxTreeNode, flag: bool):
		vartype = node.value.value
		name = node.children[0].value
		dimvar = self.interpreter_node(node.children[1]) 
		if dimvar.type == 'CELL':
			sys.stderr.write(f'error: cannot convert CELL to INT to get the dimensions count in {node.lineno}\n')
			return
		dim = int(dimvar.value)
		dimensions = self.interpreter_node(node.children[2])
		if dimensions is None or len(dimensions) != dim:
			sys.stderr.write(f'error: DIMENSIONS count shoul be equal to number of DIMENSION blocks in {node.lineno}\n')
			return
		size=1
		for i in dimensions:
			size*=i
		values=[]
		for i in range(size):
			values.append(Variable(vartype, None, flag))
		self.symbol_table[self.scope][name] = Variable(vartype, values, flag, dim, dimensions)
		
	def array_initialization(self, node: parser.SyntaxTreeNode, flag: bool):
		vartype = node.value.value
		name = node.children[0].value
		dimvar = self.interpreter_node(node.children[1]) 
		if dimvar.type == 'CELL':
			sys.stderr.write(f'error: cannot convert CELL to INT to get the dimensions count in {node.lineno}\n')
			return
		dim = int(dimvar.value)
		dimensions = self.interpreter_node(node.children[2])
		if dimensions is None or len(dimensions) != dim:
			sys.stderr.write(f'error: DIMENSIONS count shoul be equal to number of DIMENSION blocks in {node.lineno}\n')
			return
		values = self.interpreter_node(node.children[3])
		for i in range(len(values)):
			values[i] = self.type_conversion(Variable(vartype, None, False), values[i])
			if values[i] is None: 
				sys.stderr.write(f'error: initialization error of variable {name} in {node.lineno}\n')
				return
		dimension = 1
		for dmn in dimensions:
			dimension *= dmn
		if dimension != len(values):
			sys.stderr.write(f'error: dimension shoul be equal to number of <VALUE> blocks in {node.lineno}\n')
			return
		self.symbol_table[self.scope][name] = Variable(vartype, values, flag, dim, dimensions)
			
	def addition(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None): # or (len(expressions) < 2)
			sys.stderr.write(f'error: more arguments in addition expected in {op.lineno}\n')
			return
		summ = 0
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to INT to do the addition in {op.lineno}\n')
				return
			if expression.dim != 0:
				sys.stderr.write(f'error: there is no addition for arrays in {op.lineno}\n')
				return
			summ += expression.value
		return Variable('INT', summ, False)
	
	def multiplication(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None): # or (len(expressions) < 2
			sys.stderr.write(f'error: more arguments in multiplication expected in {op.lineno}\n')
			return
		mul = 1
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to INT to do the multiplication in {op.lineno}\n')
				return
			if expression.dim != 0:
				sys.stderr.write(f'error: there is no multiplication for arrays in {op.lineno}\n')
				return
			mul *= expression.value
		return Variable('INT', mul, False)
	
	def subtraction(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None) or (len(expressions) != 2):
			sys.stderr.write(f'error: two arguments in subtraction expected in {op.lineno}\n')
			return
		if (expressions[0].type == 'CELL') or (expressions[1].type == 'CELL'):
			sys.stderr.write(f'error: cannot convert CELL to INT to do the subtraction in {op.lineno}\n')
			return
		if (expressions[0].dim != 0) or (expressions[1].dim != 0):
				sys.stderr.write(f'error: there is no substraction for arrays in {op.lineno}\n')
				return
		return Variable('INT', expressions[0].value - expressions[1].value, False)

	def division(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None) or (len(expressions) != 2):
			sys.stderr.write(f'error: more arguments in division expected in {op.lineno}\n')
			return
		if expressions[1].value == 0:
			sys.stderr.write(f'error: division by zero\n')
			return
		if (expressions[0].type == 'CELL') or (expressions[1].type == 'CELL'):
			sys.stderr.write(f'error: cannot convert CELL to INT to do the division in {oplineno}\n')
			return
		if (expressions[0].dim != 0) or (expressions[1].dim != 0):
			sys.stderr.write(f'error: there is no division for arrays in {op.lineno}\n')
			return
		return Variable('INT', expressions[0].value // expressions[1].value, False)

	def maximum(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'error: more arguments in maximum expected in {op.lineno}\n')
			return
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to INT to find the maximum in {op.lineno}\n')
				return
			if expression.dim != 0:
				sys.stderr.write(f'error: there is no maximum for arrays in {op.lineno}\n')
				return
		return Variable('INT', max(expressions).value, False)

	def minimum(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'error: more arguments in minimum expected in {op.lineno}\n')
			return
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to INT to find the minimum in {op.lineno}\n')
				return
			if expression.dim != 0:
				sys.stderr.write(f'error: there is no minimum for arrays in {op.lineno}\n')
				return
		return Variable('INT', min(expressions).value, False)

	def equality(self, op: parser.SyntaxTreeNode): 
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'error: more arguments in equality expected in {op.lineno}\n')
			return
		return Variable('BOOL', len(expressions) == expressions.count(expressions[0]), False)
		
	def logic_or(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'error: more arguments in logic or expected in {op.lineno}\n')
			return
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to BOOL to do the logic or in {op.lineno}\n')
				return
			if expression.dim != 0:
				sys.stderr.write(f'error: there is no logic or for arrays in {op.lineno}\n')
				return
		return Variable('BOOL', any(expressions), False)

	def logic_and(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'error: more arguments in logic and expected in {op.lineno}\n')
			return
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to BOOL to do the logic and in {op.lineno}\n')
				return
			if expression.dim != 0:
				sys.stderr.write(f'error: there is no logic and for arrays in {op.lineno}\n')
				return
		return Variable('BOOL', all(expressions), False)
		
	def logic_not(self, op: parser.SyntaxTreeNode):
		expression = self.interpreter_node(op)
		if (expression is None):
			sys.stderr.write(f'error: more arguments in logic not expected in {op.lineno}\n')
			return
		if isinstance(expression,list):
			expression = expression[0]
		if expression.type == 'CELL':
			sys.stderr.write(f'error: cannot convert CELL to BOOL to do the logic not in {op.lineno}\n')
			return
		if expression.dim != 0:
			sys.stderr.write(f'error: there is no logic not for arrays in {op.lineno}\n')
			return
		return Variable('BOOL', not expression.value, False)

	def function_call(self, node: parser.SyntaxTreeNode):
		if self.scope == 1000:
			sys.stderr.write(f'error: recursion in {node.lineno}\n')
			return
		funcname = node.value.value
		if funcname not in self.functions.keys():
			sys.stderr.write(f'error: call of undeclarated function in {node.lineno}\n')
			return
		self.scope += 1
		self.symbol_table.append(dict())
		statements = self.functions[funcname]
		if statements is not None:
			self.interpreter_node(statements)   
		self.symbol_table.pop()
		self.scope -= 1

	def print_symbol(self):
		print(self.symbol_table)
		
if __name__ == '__main__':
	map=open('robot2','r')
	interpreter = Interpreter(map)
	f=open('path_finding','r')
	interpreter.interpreter(f.read())
	f.close()
	interpreter.print_symbol()