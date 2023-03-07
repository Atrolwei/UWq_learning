# below is the code of points to a xcoded number    
def dot2num(point,grid_size):
    return point[0] * grid_size + point[1]+1

def to_xcoded_number(points,grid_size,xcimal=16):
    xcoded_number=0
    len_points=len(points)
    for idx,point in enumerate(points):
        xcoded_number+=dot2num(point,grid_size)*xcimal**(len_points-idx-1)
    return xcoded_number

# below is the code of xcoded number to points
def num2dot(num,grid_size):
    return ((num-1)//grid_size,(num-1)%grid_size)

def to_points(xcoded_number,grid_size,xcimal=16):
    points=[]
    while xcoded_number:
        points.append(num2dot(xcoded_number%xcimal,grid_size))
        xcoded_number//=xcimal
    return points[::-1]



if __name__ == '__main__':
    points=[(0,3),(2,2),(2,1),(0,0)]
    grid_size=4
    xcimal=grid_size**2
    xcoded_number=to_xcoded_number(points,grid_size,xcimal)

    print(xcoded_number)

    points=to_points(xcoded_number,grid_size,xcimal)
    print(points)




