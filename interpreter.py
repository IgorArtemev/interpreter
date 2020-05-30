import sys
import parser
import random
from typing import List, Optional, Union
class Cell:
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.type = type

class Robot:
	def __init__(self, file, drons = 100):
		self.life = True
		s = file.readline()
		b=[]
		self.x=int(s[0])
		self.y=int(s[2])
		c=file.read()
		b=c.split('\n')
		for i in range(len(b)):
            b[i]=[int(j) for j in b[0]]
		self.map =b
		self.drons = drons
        
	def show_map(self):
		for i in range(len(self.map)):
			for j in range(len(self.map[0])):
				if i == self.x and j == self.y:
					print("0", end='') #Robot
				elif self.map[i][j] == 1: #Wall
					print("*", end='')	
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
		for cell in new_map:
			a=cell.x-self.x+5
            b=cell.y-self.y+5
            array[a*len(self.map[0])+b]=cell.type
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
		if (self.y<0 or self.y>=len(self.map)):
			self.life=False
			self.new_map.append(Cell(self.x,self.y,1))
		else:
			if self.map[self.x][self.y] == 1:
				self.life=False
			self.new_map.append(Cell(self.x,self.y,self.map[self.x][self.y]))

	def right(self):
		self.y+=1
		if (self.y<0 or self.y>=len(self.map)):
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
					self.up()
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
	
	def get_value(self,index):
		n=0
		s=len(self.value)
		for i in range(len(index)):
			s/=self.dims[i]
			n+=index[i]*s		
		return self.value[n]

	def set_value(self,index, a):
		n=0
		s=len(self.value)
		for i in range(len(index)):
			s/=self.dims[i]
			n+=index[i]*s		
		self.value[n]=a
		
	def __repr__(self):
		if self.type == 'BOOL':
			if self.value is True:
				self.value = 'TRUE'
			else:
				self.value = 'FALSE'
		return f'{self.type}, {self.value}, {self.const_flag}, {self.dim}, {self.dims}'

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

class Interpreter:

	def __init__(self):
		self.parser = parser.Parser()
		self.program = None
		self.symbol_table = [dict()]
		self.functions = dict()
		self.tree = None
		self.scope = 0

	def interpreter(self, prog=None):
		self.prog = prog
		self.tree, self.functions, parsing_ok = self.parser.parse(self.prog)
		if parsing_ok:
			#self.interpreter_tree(self.tree)
			self.interpreter_node(self.tree)
			if 'main' not in self.functions.keys():
				sys.stderr.write(f'error: no main function\n')
				return
			else:
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
		elif node.type == 'blocks': 
			for child in node.children: 
				self.interpreter_node(child)
		elif node.type == 'vardeclaration':
			self.interpreter_node(node.children)
		elif node.type == 'declarations':
			for child in node.children:
				self.interpreter_node(child)
		elif node.type == 'declaration_var':
			name = node.children.value
			if (name in self.symbol_table[self.scope].keys()) or (name in self.symbol_table[0].keys()):
				sys.stderr.write(f'error: redeclaration of variable {name}\n')
				return
			else:
				self.symbol_table[self.scope][name] = Variable(node.value.value, None, False)
		elif (node.type == 'declaration_var_init'):
			self.initialization(node, False)
		elif (node.type == 'declaration_var_const'): 
			self.initialization(node, True)
		elif (node.type == 'declaration_array'): # можно ли константы задавать другими переменными?
			name = node.children[0].value
			dim =self.interpreter_node(node.children[1]).value
			dims =self.interpreter_node(node.children[2])
			if (name in self.symbol_table[self.scope].keys()) or (name in self.symbol_table[0].keys()):
				sys.stderr.write(f'error: redeclaration of variable {name}\n')
				return
			else:
				self.symbol_table[self.scope][name] = Variable(node.value.value, None, False,dim,dims)
		elif (node.type == 'declaration_array_init'): # можно ли константы задавать другими переменными?
			self.array_initialization(node, False)
		elif (node.type == 'declaration_array_init'): # можно ли константы задавать другими переменными?
			self.array_initialization(node, True)
		elif node.type == 'function':
			if node.value.value not in self.functions.keys():
				self.functions[node.value.value] = node.children           
			else:                                                        
				sys.stderr.write(f'error: redeclaration of function {node.value.value}\n')
		elif node.type == 'function_call':
			self.function_call(node)
		elif node.type == 'statements':
			for child in node.children:
				self.interpreter_node(child)
		elif node.type == 'assignment':
			self.assignment(node)
		elif node.type == 'dimension':
			expr = self.interpreter_node(node.children) # проверить, не массив ли
			if isinstance(expr,list):
				expr = expr[0]
			if expr.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to INT to get the dimensions count\n')
				return
			return [expr.value]	
		elif node.type == 'dimensions':
			dimensions = []
			for child in node.children:
				var = self.interpreter_node(child)
				if var is not None:
					dimensions.extend(var)
				else: return None
			return dimensions
		elif node.type == 'values':
			values = []
			for child in node.children:
				var = self.interpreter_node(child)
				if var is not None:
					values.extend(var)
				else: return None
			return values
		elif node.type == 'value':
			expr = self.interpreter_node(node.children)  # проверить, не массив ли
			if isinstance(expr,list):
				expr = expr[0]
			return [expr]	
		elif node.type == 'variables':
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
			index=self.interpreter_node(node.children)
			return self.find_variable_in_array(name,index)
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
			return expressions #проверить не может ли вернуться пустой список
		elif node.type == 'conditions':
			for child in node.children:
				self.interpreter_node(child)
		elif node.type == 'condition':
			condition = self.interpreter_node(node.children['condition'])
			if condition.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to BOOL to check a condition\n')
				return
			if condition.value is True:
				self.interpreter_node(node.children['body'])
		elif node.type == 'switch':
			self.interpreter_node(node.children)
		elif node.type == 'while':
			while True:
				condition = self.interpreter_node(node.children['condition'])
				if condition.type == 'CELL':
					sys.stderr.write(f'error: cannot convert CELL to BOOL to check a condition\n')
					return
				if condition.value is True:
					self.interpreter_node(node.children['body'])
				else:
					break
		elif node.type == 'math':
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
				
	def assignment(self, node: parser.SyntaxTreeNode): 
		variables = self.interpreter_node(node.children[1])
		if variables is None:
			sys.stderr.write(f'error: incorrect variables in assignment\n')
			return
		expression = self.interpreter_node(node.children[0])
		if expression is None:
			sys.stderr.write(f'error: incorrect expression in assignment\n')
			return 
		if expression.dim ==0:
			if isinstance(expression,list):
					expression = expression[0]
			for var in variables:
				if var.const_flag is True:
					sys.stderr.write(f'error: cannot reinitialize the constant\n')
					return
				conversed_var = self.type_conversion(var, expression)
				if conversed_var is None: 
					sys.stderr.write(f'error: assignment error\n')
					return
				var.type = conversed_var.type
				var.value = conversed_var.value
		else: #массив
			for var in variables:
				if var.const_flag is True:
					sys.stderr.write(f'error: cannot reinitialize the constant\n')
					return
				if var.dims != expression.dims:
					sys.stderr.write(f'error: incorect size of array\n')
					return
				var.value=expression.value
			
	
	def find_variable(self, name):
		if name in self.symbol_table[0].keys():
			return [self.symbol_table[0][name]]
		if name in self.symbol_table[self.scope].keys():
			return [self.symbol_table[self.scope][name]]
		sys.stderr.write(f'error: undeclarated variable {name}\n')

	def find_variable_in_array(self, name, indexes):
		if name in self.symbol_table[0].keys():
			a=self.symbol_table[0][name]
		if name in self.symbol_table[self.scope].keys():
			a=self.symbol_table[self.scope][name]
		else:
			sys.stderr.write(f'error: undeclarated variable {name}\n')
			return 
		if len(indexes) != a.dim:
			sys.stderr.write(f'error: incorect count of indexes {name}\n')
			return 
		b=a.get_value(indexes)
		return Variable(a.type, b, True)
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
			sys.stderr.write(f'error: redeclaration of variable {name}\n')
			return
		expr = self.interpreter_node(node.children[1])
		if isinstance(expr,list):
			expr = expr[0]	
		var = self.type_conversion(Variable(vartype, None, flag), expr)
		if var is None: 
			sys.stderr.write(f'error: initialization error of variable {name}\n')
			return 
		self.symbol_table[self.scope][name] = var

	def array_initialization(self, node: parser.SyntaxTreeNode, flag: bool):
		vartype = node.value.value
		name = node.children[0].value
		if (name in self.symbol_table[self.scope].keys()) or (name in self.symbol_table[0].keys()):
			sys.stderr.write(f'error: redeclaration of variable {name}\n')
			return
		dimvar = self.interpreter_node(node.children[1]) #проверять, нет ли в этих expressions обращений к массиввам
		if dimvar.type == 'CELL':
			sys.stderr.write(f'error: cannot convert CELL to INT to get the dimensions count\n')
			return
		dim = dimvar.value
		dimensions = self.interpreter_node(node.children[2])
		if len(dimensions)!=dim:
			sys.stderr.write(f'error: redeclaration of variable {name}\n')
			return
		values = self.interpreter_node(node.children[3])
		for value in values:
			value = self.type_conversion(Variable(vartype, None, flag), value)
			if value is None: 
				sys.stderr.write(f'error: initialization error of variable {name}\n')
				return
			value = value.value	
		var=Variable(vartype,values,flag,dim,dimensions)
		if var is None: 
			sys.stderr.write(f'error: initialization error of variable {name}\n')
			return 
		self.symbol_table[self.scope][name] = var		
	def addition(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None): # or (len(expressions) < 2)
			sys.stderr.write(f'error: more arguments in addition expected\n')
			return
		summ = 0
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to INT to do the addition\n')
				return
			summ += expression.value
		return Variable('INT', summ, False)
	
	def multiplication(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None): # or (len(expressions) < 2
			sys.stderr.write(f'error: more arguments in multiplication expected\n')
			return
		mul = 1
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to INT to do the multiplication\n')
				return
			mul *= expression.value
		return Variable('INT', mul, False)
	
	def subtraction(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None) or (len(expressions) != 2):
			sys.stderr.write(f'error: two arguments in subtraction expected\n')
			return
		if (expressions[0].type == 'CELL') or (expressions[1].type == 'CELL'):
			sys.stderr.write(f'error: cannot convert CELL to INT to do the subtraction\n')
			return
		return Variable('INT', expressions[0].value - expressions[1].value, False)

	def division(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None) or (len(expressions) != 2):
			sys.stderr.write(f'error: more arguments in division expected\n')
			return
		if expressions[1].value == 0:
			sys.stderr.write(f'error: division by zero\n')
			return
		if (expressions[0].type == 'CELL') or (expressions[1].type == 'CELL'):
			sys.stderr.write(f'error: cannot convert CELL to INT to do the division\n')
			return
		return Variable('INT', expressions[0].value // expressions[1].value, False)

	def maximum(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'error: more arguments in maximum expected\n')
			return
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to INT to find the maximum\n')
				return
		return Variable('INT', max(expressions).value, False)

	def minimum(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'error: more arguments in minimum expected\n')
			return
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to INT to find the minimum\n')
				return
		return Variable('INT', min(expressions).value, False)

	def equality(self, op: parser.SyntaxTreeNode): #непонятно, к какому типу приводить
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'error: more arguments in equality expected\n')
			return	
		return Variable('BOOL', len(expressions) == expressions.count(expressions[0]), False)
		
	def logic_or(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'error: more arguments in logic or expected\n')
			return
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to BOOL to do the logic or\n')
				return
		return Variable('BOOL', any(expressions), False)

	def logic_and(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'error: more arguments in logic and expected\n')
			return
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'error: cannot convert CELL to BOOL to do the logic and\n')
				return
		return Variable('BOOL', all(expressions), False)
		
	def logic_not(self, op: parser.SyntaxTreeNode):
		expression = self.interpreter_node(op)
		if (expression is None):
			sys.stderr.write(f'error: more arguments in logic not expected\n')
			return
		if isinstance(expression,list):
			expression = expression[0]
		if expression.type == 'CELL':
			sys.stderr.write(f'error: cannot convert CELL to BOOL to do the logic not\n')
			return
		return Variable('BOOL', not expression.value, False)

	def function_call(self, node: parser.SyntaxTreeNode):
		if self.scope == 1000:
			sys.stderr.write(f'error: recursion\n')
			return
		funcname = node.value.value
		if funcname not in self.functions.keys():
			sys.stderr.write(f'error: call of undeclarated function\n')
			return
		self.scope += 1
		self.symbol_table.append(dict())
		statements = self.functions[funcname]
		if statements is not None:
			self.interpreter_node(statements)   
		self.scope -= 1 #хз
		self.symbol_table.pop()
		
	def print_symbol(self):
		print(self.symbol_table)
		
if __name__ == '__main__':
	interpreter = Interpreter()
	f=open('test_sorting','r')
	interpreter.interpreter(f.read())
	f.close()
	interpreter.print_symbol()