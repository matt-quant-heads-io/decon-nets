import numpy as np

#reads levels from txt file, returne 2d numpy array
def get_level(filename):
    temp = open(filename).readlines()
    level = []    
    for line in temp:
        line = line.rstrip('\n')
        row = []
        for item in line:
            if item == '.':row.append(0)
            elif item == 'b':row.append(1)
            elif item == '#':row.append(2)
            elif item == '-':row.append(3)
            elif item == 'B':row.append(4)
            elif item == 'G':row.append(5)
            elif item == 'E':row.append(6)
            elif item == 'M':row.append(0)
        level.append(row)
    return  np.asarray(level).astype(np.uint8)

#find unique patterns from target levels, returns array of unique patterns
def calc_tp(lvls, p_size):
    l = lvls[0]
    y_end = len(l)
    x_end = len(l[0])
    patterns = []
    for l in lvls:
        for (y, x), v in np.ndenumerate(l):
            if x+p_size < x_end+1 and y+p_size < y_end+1:
                pattern = str(l[y:y+p_size,x:x+p_size])
                if not (pattern in patterns):
                    patterns.append(pattern)
    return patterns

#create action space by creating unique patches from target levels
def get_actions():
    path = "./LR_Levels/"
    targets = ['Level 1.txt']
    lvls = []
    for t in targets:
        lvls.append(get_level(path + t))
    window_size = 2    
    patterns = calc_tp(lvls, window_size)
    #print(len(patterns))
    return patterns