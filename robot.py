import numpy as np
import random
class Cell:
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.type = type

class Robot:
    def __init__(self, x, y, _map, drons = 100):
        self.life = True
        self.x = x
        self.y = y
        self.map = _map
        self.drons = drons
        
    def show_map(self):
        for row in self.map:
            for cell in row:
                if cell == 1: #Wall
                    print("*", end='')
                elif cell ==3: #Exit
                    print("X", end='')
                else:          #Empty
                    print(" ", end='')
            print()
    
    def up(self, n):
        for i in range(n):
            if self.life:
                self.y+=1
                if self.map[self.x][self.y] == 1:
                    self.life=False

    def down(self, n):
        for i in range(n):
            if self.life:
                self.y-=1
                if self.map[self.x][self.y] == 1:
                    self.life=False

    def left(self, n):
        for i in range(n):
            if self.life:
                self.x-=1
                if self.map[self.x][self.y] == 1:
                    self.life=False

    def right(self, n):
        for i in range(n):
            if self.life:
                self.x+=1
                if self.map[self.x][self.y] == 1:
                    self.life=False

    def drons_count(self):
        return self.drons
    
    def send_drons(self, n):
        self.drons-=n
        new_map=[]
        for i in range(n):
            a=Satellite(self.x, self.y, self.map)
            new_map.append(a.exploring())    
        min_x = new_map[0].x
        min_y = new_map[0].y
        max_x=min_x
        max_y=min_y
        for cell in new_map:
            if cell.x > max_x:
                max_x=cell.x
            if cell.x < min_x:
                min_x=cell.x
            if cell.y > max_y:
                max_y=cell.y
            if cell.y < min_y:
                min_y=cell.y
        

    
class Satellite:
    def __init__(self, x, y, _map):
        self.life = True
        self.x = x
        self.y = y
        self.map = _map
        self.new_map = []
    
    def up(self):
        self.y+=1
        if self.map[self.x][self.y] == 1:
            self.life=False
        self.new_map.append(Cell(self.x,self.y,self.map[self.x][self.y]))

    def down(self):
        self.y-=1
        if self.map[self.x][self.y] == 1:
                self.life=False
        self.new_map.append(Cell(self.x,self.y,self.map[self.x][self.y]))

    def left(self):
        self.x-=1
        if self.map[self.x][self.y] == 1:
            self.life=False
        self.new_map.append(Cell(self.x,self.y,self.map[self.x][self.y]))

    def right(self):
        self.x+=1
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