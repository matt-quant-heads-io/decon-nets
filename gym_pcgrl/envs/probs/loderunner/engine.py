#import
import numpy as np
from PIL import Image
import os
import pandas as pd
import random
from queue import PriorityQueue
import heapq
import random
import copy
import time
import sys


class Map2D:
    def __init__(self,data):
        self.w=len(data[0])  #cols
        self.h=len(data)     #rows
        self.data=np.asarray(data)

    def __getitem__(self, item):
        return self.data[item]
    
    def replace(self, x,y,a):
        self.data[x,y] = a
        
    

class Block:
    def __init__(self,row,col):
        self.row = row
        self.col = col
        
         
class Node:
    def __init__(self,row,col,map2d,block,action,golds,parent=None):
        self.row = row
        self.col = col
        self.parent = parent
        self.level = map2d
        self.action = action
        self.golds = golds
        
        if self.parent!= None:
            self.step = self.parent.step+1
            self.blocks = copy.deepcopy(self.parent.blocks)
            for i,b in enumerate(self.blocks):
                #b.ticks -= 1
                row_diff = self.row - b.row
                col_diff = abs(self.col - b.col)
                
                if row_diff >= 1 or col_diff > 3:
                    self.level.replace(b.row,b.col,'b')
                    del self.blocks[i]
            if block != None:
                self.blocks.append(block)          
        else:
            self.step = 0
            self.blocks = list()  
        self.score = -1
        
    def get_key(self):
        result = ""
        for x in range(self.level.h):
            for y in range(self.level.w):
                result += str(self.level[x][y])
        return "{},{},{}".format(result,self.row,self.col)        
        
    def get_score(self,golds):
        rem_golds = [g for g in golds if g not in self.golds] 
        gold_dist = [0]
        for g in rem_golds:
            dist = abs(self.row-g[0]) + abs(self.col-g[1])
            gold_dist.append(dist)
            
        if self.score == -1:
            self.score =  min(gold_dist) - len(self.golds)
        return self.score
        
    #returns next valid actions
    def get_actions(self):
        level = self.level
        left_end = 0
        right_end = self.level.w-1
        top = 0
        bottom = self.level.h-1
        row = self.row
        col = self.col
        actions = []
        prev_act = self.action
        #if previous action is dig
        if prev_act=="dig-left" or prev_act=="right-dig-left":
            if col!=right_end and level[row+1,col]=='b' and (level[row,col]=='.' or level[row,col]=='G') and level[row,col+1]!='b' and level[row,col+1]!='B' and(
            level[row+1,col+1]=='B' or level[row+1,col+1]=='b' or level[row+1,col+1]=='#'):actions.append("right-dig-left")
            if row!=bottom and col!=left_end:actions.append("d-left")
        elif prev_act=="dig-right" or prev_act=="left-dig-right":
            if col!=left_end and level[row+1,col]=='b' and (level[row,col]=='.' or level[row,col]=='G') and level[row,col-1]!='b' and level[row,col-1]!='B' and(
            level[row+1,col-1]=='B' or level[row+1,col-1]=='b' or level[row+1,col-1]=='#'):actions.append("left-dig-right")
            if row!=bottom and col!=right_end:actions.append("d-right")

        else:  
            #if player is on the lowest row  
            if row==bottom:
                if col!=left_end and level[row,col-1]!='b' and level[row,col-1]!='B': actions.append("left")
                if col!=right_end and level[row,col+1]!='b' and level[row,col+1]!='B': actions.append("right")
                if level[row,col]=='#' and level[row-1,col]!='b' and level[row-1,col]!='B': actions.append("up")

            #if current position is ladder or rope
            elif level[row,col] == '#' or level[row,col] == '-':
                #if player is not on the lowest row
                if row!=bottom:                    
                    if level[row,col]=='#' and row!=top and level[row-1,col]!='b' and level[row-1,col]!='B': actions.append("up")
                    if level[row+1,col]!='b' and level[row+1,col]!='B': actions.append("down")
                    if col!=left_end and (level[row,col-1]=='#' or level[row,col-1]=='-'): actions.append("left")
                    if col!=left_end and (level[row,col-1]=='G' or level[row,col-1]=='.') and (
                       level[row+1,col-1]=='b' or level[row+1,col-1]=='B' or level[row+1,col-1]=='#'): actions.append("left")
                    if col!=right_end and (level[row,col+1]=='#' or level[row,col+1]=='-'): actions.append("right")  
                    if col!=right_end and (level[row,col+1]=='G' or level[row,col+1]=='.') and (
                        level[row+1,col+1]=='b' or level[row+1,col+1]=='B' or level[row+1,col+1]=='#'): actions.append("right")
                    if col!=left_end and (level[row,col-1]=='G' or level[row,col-1]=='.') and (
                        level[row+1,col-1]!='b' and level[row+1,col-1]!='B' and level[row+1,col-1]!='#'): actions.append("d-left")
                    if col!=right_end and (level[row,col+1]=='G' or level[row,col+1]=='.') and (
                        level[row+1,col+1]!='b' and level[row+1,col+1]!='B' and level[row+1,col+1]!='#'): actions.append("d-right")
                    if col!=left_end and row!=bottom-1 and level[row+1,col-1]=='b' and (
                        level[row,col-1]=='.' or level[row,col-1]=='G'):actions.append("dig-left")
                    if col!=right_end and row!=bottom-1 and level[row+1,col+1]=='b' and (
                        level[row,col+1]=='.' or level[row,col+1]=='G'):actions.append("dig-right")

            #if current position is empty or gold or enemy 
            elif level[row,col]=='.' or level[row,col]=='G':
                #if player is not on the lowest row
                if row!=bottom:
                    #below is empty or rope or gold
                    if level[row+1,col]!='b' and level[row+1,col]!='B':actions.append("down")

                    #below is block or ladder
                    if level[row+1,col] =='b' or level[row+1,col] =='B' or level[row+1,col] =='#':
                        if col!=left_end and (level[row,col-1]=='#' or level[row,col-1]=='-'): actions.append("left")
                        if col!=left_end and (level[row,col-1]=='G' or level[row,col-1]=='.') and (
                            level[row+1,col-1]=='b' or level[row+1,col-1]=='B' or level[row+1,col-1]=='#'): actions.append("left")
                        if col!=right_end and (level[row,col+1]=='#' or level[row,col+1]=='-'): actions.append("right")  
                        if col!=right_end and (level[row,col+1]=='G' or level[row,col+1]=='.') and (
                            level[row+1,col+1]=='b' or level[row+1,col+1]=='B' or level[row+1,col+1]=='#'): actions.append("right")
                        if col!=left_end and (level[row,col-1]=='G' or level[row,col-1]=='.') and (
                            level[row+1,col-1]!='b' and level[row+1,col-1]!='B' and level[row+1,col-1]!='#'): actions.append("d-left")
                        if col!=right_end and (level[row,col+1]=='G' or level[row,col+1]=='.') and (
                            level[row+1,col+1]!='b' and level[row+1,col+1]!='B' and level[row+1,col+1]!='#'): actions.append("d-right")
                        if col!=left_end and level[row+1,col-1]=='b' and row!=bottom-1 and(level[row,col-1]=='.' 
                            or level[row,col-1]=='G'):actions.append("dig-left")
                        if col!=right_end and level[row+1,col+1]=='b'and row!=bottom-1 and(level[row,col+1]=='.' 
                            or level[row,col+1]=='G'):actions.append("dig-right")

        return actions    
    
    #returns children of a node
    def get_children(self):
        child_nodes = []
        valid_actions = self.get_actions()
        #for each action create a child node
        for i in range(len(valid_actions)):
            action = valid_actions[i]
            #next position based on action
            if action == 'left':
                child_row = self.row
                child_col = self.col-1
                child_level = copy.deepcopy(self.level)
                block = None
            elif action == 'right':
                child_row = self.row
                child_col = self.col+1
                child_level = copy.deepcopy(self.level)
                block = None
            elif action == 'up':
                child_row = self.row-1
                child_col = self.col
                child_level = copy.deepcopy(self.level)
                block = None
            elif action == 'down':
                child_row = self.row+1
                child_col = self.col
                child_level = copy.deepcopy(self.level)
                block = None
            elif action == 'd-left':
                child_row = self.row+1
                child_col = self.col-1
                child_level = copy.deepcopy(self.level)
                block = None
            elif action == 'd-right':
                child_row = self.row+1
                child_col = self.col+1
                child_level = copy.deepcopy(self.level)
                block = None
            elif action == 'dig-left':
                child_row = self.row
                child_col = self.col
                child_level = copy.deepcopy(self.level)
                child_level.replace(child_row+1,child_col-1,'.') 
                block = Block(child_row+1,child_col-1)
            elif action == 'dig-right':
                child_row = self.row
                child_col = self.col
                child_level = copy.deepcopy(self.level)
                child_level.replace(child_row+1,child_col+1,'.')  
                block = Block(child_row+1,child_col+1)
            elif action == 'left-dig-right':
                child_row = self.row
                child_col = self.col-1
                child_level = copy.deepcopy(self.level)
                child_level.replace(child_row+1,child_col+1,'.')  
                block = Block(child_row+1,child_col+1)
            elif action == 'right-dig-left':
                child_row = self.row
                child_col = self.col+1
                child_level = copy.deepcopy(self.level)
                child_level.replace(child_row+1,child_col-1,'.') 
                block = Block(child_row+1,child_col-1)  
            else:  
                print("error")  
            
            golds = copy.deepcopy(self.golds)
            if child_level[child_row,child_col] == 'G' and (child_row,child_col) not in golds:
                child_level.replace(child_row,child_col,'.')
                golds.append((child_row,child_col))
 
            blocks = copy.deepcopy(self.blocks)
           
            #create child node
            child = Node(child_row,child_col,child_level,block,action,golds,self)
            child_nodes.append(child)
            
        return child_nodes

    
    def get_path(self,root):
        row = root.row
        col = root.col
        lvl = root.level
        nodes = list()
        path = list()
        node = self
        while node.row!= row or node.col!= col or node.level!=lvl:
            if (node.row, node.col) not in path:
                path.append((node.row, node.col))
            nodes.append(node)
            node = node.parent
        path.append((node.row, node.col))
        nodes.reverse()
        return path, nodes
    
    def get_dist_score(self,golds):  
        golds_dist = {}
        for g in golds:
            dist = abs(self.row-g[0]) + abs(self.col-g[1])
            golds_dist.update({g: dist})
        return golds_dist
    
    
class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority,priority2):
        heapq.heappush(self.elements,(priority,priority2,item))
    
    def get(self):
        return heapq.heappop(self.elements)[2]
    
    def display(self):
        for i in range (len(self.elements)):
            print(self.elements[i][2].row , self.elements[i][2].col,self.elements[i][2].score,'--',self.elements[i][0],self.elements[i][1],self.elements[i][2] )           

def find_golds_astar(root,golds_list):
    max_dist = root.level.w+root.level.h
    gold_dist = {}
    for g in golds_list:
        gold_dist.update({g: max_dist})
        
    start_time = time.time()
    all_nodes = 0
    best = root
    max_gold = 0
    score = root.get_score(golds_list)
    queue = PriorityQueue()
    #push node in the list
    queue.put(root,score,all_nodes)
    visited = set()

    #while the list is not empty
    while not queue.empty() and  time.time()-start_time < 1:
        current = queue.get()
        if(current.get_key() not in visited):
            new_dist = current.get_dist_score(golds_list)
            for k in gold_dist.keys():
                if k in new_dist.keys():
                    if new_dist[k] < gold_dist[k]:
                        gold_dist[k] = new_dist[k]
                        
            #check for goal condition
            if len(current.golds) == len(golds_list):
                #print('found',checked_nodes,'/',all_nodes)
                #print(time.time()-start_time)
                return current,gold_dist
              
            visited.add(current.get_key())
            if len(current.golds) > max_gold:
                max_gold = len(current.golds)
                best = current
              
            children = current.get_children()
            for c in (children):
                all_nodes += 1
                score = c.get_score(golds_list)
                queue.put(c,score,all_nodes)
    #print('not found',checked_nodes,'/',all_nodes)
    #print(time.time()-start_time)
    return best,gold_dist

def get_golds(level):
    golds = []
    for i in range(level.h):
        for j in range(level.w):
            if level[i,j]=='G':
                golds.append((i,j))
    return golds


def get_starting_point(map2d):
    row=0
    col=0
    for i in range(map2d.h):
        for j in range(map2d.w):
            if map2d[i,j]=='M':
                row=i
                col=j
            elif map2d[i,j]=='E':
                map2d.replace(i,j,'.')
    return row,col

# to check playability run this(level should have a player)
# input: dir, file
# returns collected golds, no of total golds, time
def AStarAgent(level):
    map2d = Map2D(level)
    row,col = get_starting_point(map2d)
    golds = get_golds(map2d)
    map2d.replace(row,col,'.')
    root = Node(row,col,map2d,None,None,[],None)
    collected,gold_dists = find_golds_astar(root,golds)
    path,_ = collected.get_path(root)
    dist_to_golds = 0
    for k in gold_dists.keys():
        if k not in collected.golds:
            dist_to_golds += gold_dists[k]
    dist = dist_to_golds/(map2d.h + map2d.w)
  
    return len(collected.golds), dist, len(path)


