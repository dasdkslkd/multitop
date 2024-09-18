import mat73 as sio
import pyopenvdb as vdb
import numpy as np

def writeObjFile(filename, points, triangles=[], quads=[]):
    f = open(filename, 'w')
    # Output vertices.
    for xyz in points:
        f.write('v %g %g %g\n' % tuple(xyz))
    f.write('\n')
    # Output faces.
    for ijk in triangles:
        f.write('f %d %d %d\n' %
(ijk[0]+1, ijk[1]+1, ijk[2]+1)) # offset vertex indices by one
    for ijkl in quads:
        f.write('f %d %d %d %d\n' %
(ijkl[0]+1, ijkl[1]+1, ijkl[2]+1, ijkl[3]+1))
    f.close()

# 从MAT文件中加载矩阵数据
# mat_data = sio.loadmat('x_1_6_y_1_6_z_1_18_all.mat')
mat_data = sio.loadmat('x_1_10_y_1_10_z_1_2_all.mat')
matrix = mat_data['Xsys']
#print(matrix.shape)

#sio.savemat('file_name.mat', {'data':matrix}) 
# 将加载的矩阵转换为NumPy数组
#np_array = np.array(matrix)
np_array=matrix.astype(float)
#print(np.where(np_array==False))
#quit()
shape=np_array.shape
mylen=np.max(shape)
new_array=np_array
new_array=np.zeros([mylen,mylen,mylen],dtype=float)

new_array[0:shape[0],0:shape[1],0:shape[2]]=np_array
print(new_array.shape)
# 创建一个空的FloatGrid对象
grid = vdb.FloatGrid()
# 将NumPy数组的数据复制到Grid对象
grid.copyFromArray(new_array)

# 保存Grid对象为VDB文件
filename = "output.vdb"
vdb.write(filename, grid)
