import sys
import copy
import parser
from typing import List, Optional, Union
from robot import Robot
from variable import Variable

class Interpreter:

	def __init__(self,file):
		self.parser = parser.Parser()
		self.program = None
		self.variable_table = [dict()]
		self.functions_table_table = dict()
		self.tree = None
		self.scope = 0
		self.robot=Robot(file)

	def interpreter(self, prog=None):
		self.prog = prog
		self.tree, self.functions_table, parsing_ok = self.parser.parse(self.prog)
		if parsing_ok:
			self.interpreter_node(self.tree)
			if 'main' not in self.functions_table.keys():
				sys.stderr.write(f'ERROR: no main function\n ')
				return
			else:
				self.scope=1
				self.variable_table.append(dict())
				self.interpreter_node(self.functions_table['main'])
				
		else:
			sys.stderr.write(f'Can\'t intemperate this, incorrect syntax\n')

	def interpreter_tree(self, tree):
		print("Program tree:\n")
		tree.print()
		print("\n")

	def print_symbol(self):
		print(self.variable_table)

	def initialization(self, node: parser.SyntaxTreeNode, flag: bool):
		type = node.value.value
		name = node.children[0].value
		if (name in self.variable_table[self.scope].keys()) or (name in self.variable_table[0].keys()):
			sys.stderr.write(f'ERROR: redeclaration of variable {name} in {node.lineno}\n')
			return
		expression = self.interpreter_node(node.children[1])
		if isinstance(expression,list):
			exprression = expr[0]	
		variable = self.type_conversion(Variable(type, None, flag), expression)
		if variable is None: 
			sys.stderr.write(f'ERROR: initialization error of variable {name} in {node.lineno}\n')
			return 
		self.variable_table[self.scope][name] = variable

	def array_declaration(self, node: parser.SyntaxTreeNode, flag: bool):
		type = node.value.value
		name = node.children[0].value
		dim_var = self.interpreter_node(node.children[1]) 
		if dim_var.type == 'CELL':
			sys.stderr.write(f'ERROR: cannot convert CELL to INT  in {node.lineno}\n')
			return
		dim = int(dim_var.value)
		dimensions = self.interpreter_node(node.children[2])
		if dimensions is None or len(dimensions) != dim:
			sys.stderr.write(f'ERROR: DIMENSIONS count should be equal to number of DIMENSION blocks in {node.lineno}\n')
			return
		size=1
		for i in dimensions:
			size*=i
		values=[]
		for i in range(size):
			values.append(Variable(type, None, flag))
		self.variable_table[self.scope][name] = Variable(type, values, flag, dim, dimensions)
		
	def array_initialization(self, node: parser.SyntaxTreeNode, flag: bool):
		type = node.value.value
		name = node.children[0].value
		dim_var = self.interpreter_node(node.children[1]) 
		if dim_var.type == 'CELL':
			sys.stderr.write(f'ERROR: cannot convert CELL to INT to get the dimensions count in {node.lineno}\n')
			return
		dim = int(dim_var.value)
		dimensions = self.interpreter_node(node.children[2])
		if dimensions is None or len(dimensions) != dim:
			sys.stderr.write(f'ERROR: DIMENSIONS count shoul be equal to number of DIMENSION blocks in {node.lineno}\n')
			return
		values = self.interpreter_node(node.children[3])
		for i in range(len(values)):
			values[i] = self.type_conversion(Variable(type, None, False), values[i])
			if values[i] is None: 
				sys.stderr.write(f'ERROR: initialization error of variable {name} in {node.lineno}\n')
				return
		dimension = 1
		for i in dimensions:
			dimension *= i
		if dimension != len(values):
			sys.stderr.write(f'ERROR: dimension should be equal to number of VALUE blocks in {node.lineno}\n')
			return
		self.variable_table[self.scope][name] = Variable(type, values, flag, dim, dimensions)
	def find_variable(self, name):
		if name in self.variable_table[0].keys():
			return [self.variable_table[0][name]]
		if name in self.variable_table[self.scope].keys():
			return [self.variable_table[self.scope][name]]
		sys.stderr.write(f'ERROR: undeclarated variable {name}\n')
		
	def find_variable_in_array(self, name, indexes: list):
		var = self.find_variable(name)
		if var[0].dim == 0: 
			sys.stderr.write(f'ERROR: variable {name} is not an array\n')
			return
		if var[0].dim != len(indexes): 
			sys.stderr.write(f'ERROR: incorrect count of indexes\n')
			return
		for a in range(len(indexes)):
			if indexes[a]>=var[0].dims[a]:
				sys.stderr.write(f'ERROR: out of the range \n')
				return
		i = var[0].count_of_offset(indexes)
		if var[0].const is True:
			var[0].value[i].const=True
		return [var[0].value[i]]
		
	def assignment(self, node: parser.SyntaxTreeNode): 
		variables = self.interpreter_node(node.children[1])
		if variables is None:
			sys.stderr.write(f'ERROR: incorrect variables  in {node.lineno}\n')
			return
		expression = self.interpreter_node(node.children[0])
		if expression is None:
			sys.stderr.write(f'ERROR: incorrect expression  in {node.lineno}\n')
			return 
		if isinstance(expression,list):
			expression = expression[0]
		a = 0
		for var in variables:
			if var.dim != 0:
				a = a + 1
		if expression.dim == 0 and a == 0:
			for var in variables:
				if var.const is True:
					sys.stderr.write(f'ERROR: cannot reinitialize the constant in {node.lineno}\n')
					return
				conversed_var = self.type_conversion(var, expression)
				if conversed_var is None: 
					sys.stderr.write(f'ERROR: assignment error in {node.lineno}\n')
					return
				var.type = conversed_var.type
				var.value = conversed_var.value
		elif expression.dim != 0 and a == len(variables) and a != 0: 
			for var in variables:
				if var.const is True:
					sys.stderr.write(f'ERROR: cannot reinitialize the constant in {node.lineno}\n')
					return
				if var.dims != expression.dims:
					sys.stderr.write(f'ERROR: different sizes of arrays in assignment in {node.lineno}\n')
					return
				var.value = copy.deepcopy(expression.value)	
		else:
			sys.stderr.write(f'ERROR: cannot assign array to not array and vice versa in {node.lineno}\n')
			return
	
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
		var2.const = var1.const
		if var1.type != var2.type:
			if (var1.type == 'CELL') or (var2.type == 'CELL'):
				sys.stderr.write(f'ERROR: cannot convert {var2.type} to {var1.type}\n')
				return
			elif var2.type == 'BOOL':
				return self.bool_to_int(var2)
			elif var2.type == 'INT':
				return self.int_to_bool(var2)
		else:
			return var2
			
	def addition(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if expressions is None:
			sys.stderr.write(f'ERROR: no arguments in addition in {op.lineno}\n')
			return
		summ = 0
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'ERROR: cannot convert CELL to INT to do the addition\n')
				return
			if expression.dim != 0:
				sys.stderr.write(f'ERROR: there is no addition for arrays\n')
				return
			if expression.value == None:
				sys.stderr.write(f'ERROR: uninitialized variable\n')
				return
			summ += expression.value
		return Variable('INT', summ, False)
	
	def multiplication(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if expressions is None: 
			sys.stderr.write(f'ERROR: no arguments in multiplication  in {op.lineno}\n')
			return
		mul = 1
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'ERROR: cannot convert CELL to INT to do the multiplication in {op.lineno}\n')
				return
			if expression.dim != 0:
				sys.stderr.write(f'ERROR: there is no multiplication for arrays in {op.lineno}\n')
				return
			if expression.value == None:
				sys.stderr.write(f'ERROR: uninitialized variable\n')
				return
			mul *= expression.value
		return Variable('INT', mul, False)
	
	def subtraction(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None) or (len(expressions) != 2):
			sys.stderr.write(f'ERROR: two arguments in subtraction expected in {op.lineno}\n')
			return
		if (expressions[0].type == 'CELL') or (expressions[1].type == 'CELL'):
			sys.stderr.write(f'ERROR: cannot convert CELL to INT to do the subtraction in {op.lineno}\n')
			return
		if (expressions[0].dim != 0) or (expressions[1].dim != 0):
				sys.stderr.write(f'ERROR: there is no substraction for arrays in {op.lineno}\n')
				return
		if  expressions[0].value == None or expressions[1].value == None:
				sys.stderr.write(f'ERROR: uninitialized variable\n')
				return
		return Variable('INT', expressions[0].value - expressions[1].value, False)

	def division(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None) or (len(expressions) != 2):
			sys.stderr.write(f'ERROR: more arguments in division expected in {op.lineno}\n')
			return
		if expressions[1].value == 0:
			sys.stderr.write(f'ERROR: division by zero\n')
			return
		if (expressions[0].type == 'CELL') or (expressions[1].type == 'CELL'):
			sys.stderr.write(f'ERROR: cannot convert CELL to INT to do the division in {op.lineno}\n')
			return
		if (expressions[0].dim != 0) or (expressions[1].dim != 0):
			sys.stderr.write(f'ERROR: there is no division for arrays in {op.lineno}\n')
			return
		if  expressions[0].value == None or expressions[1].value == None:
				sys.stderr.write(f'ERROR: uninitialized variable\n')
				return
		return Variable('INT', expressions[0].value // expressions[1].value, False)

	def maximum(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'ERROR: no arguments in maximum  in {op.lineno}\n')
			return
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'ERROR: cannot convert CELL to INT to find the maximum in {op.lineno}\n')
				return
			if expression.dim != 0:
				sys.stderr.write(f'ERROR: there is no maximum for arrays in {op.lineno}\n')
				return
		return Variable('INT', max(expressions).value, False)

	def minimum(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'ERROR: no arguments in minimum in {op.lineno}\n')
			return
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'ERROR: cannot convert CELL to INT to find the minimum in {op.lineno}\n')
				return
			if expression.dim != 0:
				sys.stderr.write(f'ERROR: there is no minimum for arrays in {op.lineno}\n')
				return
			if expression.value == None:
				sys.stderr.write(f'ERROR: uninitialized variable\n')
				return	
		return Variable('INT', min(expressions).value, False)

	def equality(self, op: parser.SyntaxTreeNode): 
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'ERROR:no arguments in equality in {op.lineno}\n')
			return
		return Variable('BOOL', len(expressions) == expressions.count(expressions[0]), False)
		
	def logic_or(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'ERROR: no arguments in logic or  in {op.lineno}\n')
			return
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'ERROR: cannot convert CELL to BOOL to do the logic or in {op.lineno}\n')
				return
			if expression.dim != 0:
				sys.stderr.write(f'ERROR: there is no logic or for arrays in {op.lineno}\n')
				return
			if expression.value == None:
				sys.stderr.write(f'ERROR: uninitialized variable\n')
				return
		return Variable('BOOL', any(expressions), False)

	def logic_and(self, op: parser.SyntaxTreeNode):
		expressions = self.interpreter_node(op)
		if (expressions is None):
			sys.stderr.write(f'ERROR: no arguments in logic and  in {op.lineno}\n')
			return
		for expression in expressions:
			if expression.type == 'CELL':
				sys.stderr.write(f'ERROR: cannot convert CELL to BOOL to do the logic and in {op.lineno}\n')
				return
			if expression.dim != 0:
				sys.stderr.write(f'ERROR: there is no logic and for arrays in {op.lineno}\n')
				return
			if expression.value == None:
				sys.stderr.write(f'ERROR: uninitialized variable\n')
				return
		return Variable('BOOL', all(expressions), False)
		
	def logic_not(self, op: parser.SyntaxTreeNode):
		expression = self.interpreter_node(op)
		if (expression is None):
			sys.stderr.write(f'ERROR: no arguments in logic not  in {op.lineno}\n')
			return
		if isinstance(expression,list):
			expression = expression[0]
		if expression.type == 'CELL':
			sys.stderr.write(f'ERROR: cannot convert CELL to BOOL to do the logic not in {op.lineno}\n')
			return
		if expression.dim != 0:
			sys.stderr.write(f'ERROR: there is no logic not for arrays in {op.lineno}\n')
			return
		if expression.value == None:
				sys.stderr.write(f'ERROR: uninitialized variable\n')
				return
		return Variable('BOOL', not expression.value, False)

	def function_call(self, node: parser.SyntaxTreeNode):
		if self.scope == 5000:
			sys.stderr.write(f'ERROR: recursion in {node.lineno}\n')
			return
		funcname = node.value.value
		if funcname not in self.functions_table.keys():
			sys.stderr.write(f'ERROR: call of undeclarated function in {node.lineno}\n')
			return
		self.scope += 1
		self.variable_table.append(dict())
		statements = self.functions_table[funcname]
		if statements is not None:
			self.interpreter_node(statements)   
		self.variable_table.pop()
		self.scope -= 1

	def interpreter_node(self, node: parser.SyntaxTreeNode):
		if node is None:
			return
		if node.type == 'program':
			self.interpreter_node(node.children)
		elif node.type == 'function':
			if node.value.value not in self.functions_table.keys():
				self.functions_table[node.value.value] = node.children 
			else:                                                        
				sys.stderr.write(f'ERROR: redeclaration of function {node.value.value} in {node.lineno}\n')
		elif node.type == 'function_call': 
			self.function_call(node)
		elif node.type == 'blocks' or node.type == 'conditions' or node.type == 'statements' or node.type == 'declarations':
			for child in node.children: 
				self.interpreter_node(child)
		elif node.type == 'vardeclaration' or node.type == 'switch':
			self.interpreter_node(node.children)
		elif node.type == 'declaration_var':
			name = node.children.value
			if (name in self.variable_table[self.scope].keys()) or (name in self.variable_table[0].keys()):
				sys.stderr.write(f'ERROR: redeclaration of variable {name} in {node.lineno}\n')
				return
			else:
				self.variable_table[self.scope][name] = Variable(node.value.value, None, False)
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
		elif node.type == 'variables' or node.type == 'dimensions' or node.type == 'indexes' or node.type == 'values': 
			variables = []
			for child in node.children:
				var = self.interpreter_node(child)
				if var is not None:
					variables.extend(var)
				else: return None
			return variables
		elif node.type == 'assignment':
			self.assignment(node)
		elif node.type == 'variable':
			name = node.value.value
			return self.find_variable(name)
		elif node.type == 'variable_in_array':
			name = node.value.value
			indexes = self.interpreter_node(node.children)
			return self.find_variable_in_array(name, indexes)
		elif node.type == 'dimension':
			expr = self.interpreter_node(node.children)
			if isinstance(expr,list):
				expr = expr[0]
			if expr.dim != 0:
				sys.stderr.write(f'ERROR: dimension count cannot be initialized with array in {node.lineno}\n')
				return
			if expr.type == 'CELL':
				sys.stderr.write(f'ERROR: cannot convert CELL to INT to get the dimensions count in {node.lineno}\n')
				return
			return [int(expr.value)]
		elif node.type == 'index':
			expr = self.interpreter_node(node.children) 
			if isinstance(expr,list):
				expr = expr[0]
			if expr.type == 'CELL':
				sys.stderr.write(f'ERROR: cannot convert CELL to INT to get the index in {node.lineno}\n')
				return
			if expr.dim != 0:
				sys.stderr.write(f'ERROR: index cannot be initialized with array in {node.lineno}\n')
				return
			return [int(expr.value)]
		elif node.type == 'value':
			expr = self.interpreter_node(node.children)
			if isinstance(expr,list):
				expr = expr[0]
			if expr.dim != 0:
				sys.stderr.write(f'ERROR: value cannot be initialized with array in {node.lineno}\n')
				return
			return [expr]	
		elif node.type == 'const':
			value = node.value
			if isinstance(value, int):
				return Variable('INT', value, True)
			elif value == 'TRUE':
				return Variable('BOOL', True, True)
			elif value == 'FALSE':
				return Variable('BOOL', False, True)
			else:
				return Variable('CELL', value, True) #EMPTY/WALL/EXIT/UNDEF
		elif node.type == 'expressions':			
			expressions = [] 
			for child in node.children:
				expression = self.interpreter_node(child)
				if expression is None:
					return None
				else: 
					if isinstance(expression,list):
						expressions.extend(expression)
					else:
						expressions.append(expression)
			return expressions 
		elif node.type == 'condition':
			condition = self.interpreter_node(node.children['condition'])
			if isinstance(condition,list):
				condition = condition[0]
			if condition.dim != 0:
				sys.stderr.write(f'ERROR: array cannot be condition in {node.lineno}\n')
				return
			if condition.type == 'CELL':
				sys.stderr.write(f'ERROR: cannot convert CELL to BOOL to check a condition in {node.lineno}\n')
				return
			if bool(condition.value) is True:
				self.interpreter_node(node.children['body'])
		elif node.type == 'while':
			while True:
				condition = self.interpreter_node(node.children['condition'])
				if isinstance(condition,list):
					condition = condition[0]
				if condition.dim != 0:
					sys.stderr.write(f'ERROR: array cannot be condition in {node.lineno}\n')
					return
				if condition.type == 'CELL':
					sys.stderr.write(f'ERROR: cannot convert CELL to BOOL to check a condition in {node.lineno}\n')
					return
				if bool(condition.value) is True:
					self.interpreter_node(node.children['body'])
				else:
					break
		elif node.type == 'standart_function': #проверять expression на немассивность
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
				count=self.robot.drons_count()
				if var.dim !=0 or var.type != 'INT':
					sys.stderr.write(f'ERROR: assignment error in {node.lineno}\n')
					return
				var.value=count
		elif node.type == 'senddrons':
			n=self.interpreter_node(node.children)
			map=self.robot.send_drons(n.value)
			cells=[]
			cell=['UNDEF','WALL','EMPTY','EXIT']
			for i in map:
				cells.append(Variable('CELL', cell[i], False))
			return Variable('CELL', cells, True, 2, [11,11])

		
if __name__ == '__main__':
	map=open('test_robot','r')
	interpreter = Interpreter(map)
	f=open('path_finding','r')
	interpreter.interpreter(f.read())
	f.close()
	interpreter.print_symbol()