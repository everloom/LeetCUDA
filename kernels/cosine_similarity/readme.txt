A100上每个sm最大寄存器数量为65536个32bit的寄存器，最大smem为164kb(167936 byte)，sm数量为108
L20上每个sm最大寄存器数量为65536个32bit的寄存器，最大smem为100kb(102400 byte)，sm数量为92

v2相比v1，去掉了那些warp shuffle外面的if (laneid % 4 == 0)的条件判断
###########################################################################################
v3改成了a矩阵8*128，b矩阵250000*128的形式，祥总的建议，首先a矩阵放到寄存器里面，注意这么几点：
1、tensorcore算力要打满
2、重点关注L2 cache
3、所有的sm上的block尽量处理同一行的数据（这个说实话有点没搞懂啥意思）
4、b矩阵到底是normal还是transpose的layout这个祥总还没说

1、用m8n8k4的mma，然后一次算一个8*16和8*16的mma，得到8*8的结果
一次cp async的大小是8*16 启动block数量为250000/8=31250，slice k的大小为16。每个block都会重复从A和B的gemm载入数据，每个block计算8次mma，其中mainloop占了可能就只占4 5次左右
这种实现的单个block的reg大小为(单位为byte)：
A：8*16*2 B:8*16*2 C:8*8*2，总的需要uint32寄存器个数: (8*16*2 + 8*16*2 + 8*8*2)/4=160
单个block smem大小:
A: stage*8*16*2 B:stage*8*16*2 总smem大小(byte,stage为3):1536 byte
min(65536/160,167936/1536) = min(409.6,109.3)=109.3
每个sm最多容许109.3个block

本人评价：这种方案对于A和B，每个block要发射总共16次cp async（a和b各占8次），每次cp async只读8*16个half，读取的过于小了，且mainloop调用mma指令次数较少，感觉效率可能不高

2、用m16n8k16的mma计算，对于A矩阵，一次cp async将整个8*128的矩阵载入，对于B矩阵，一次载入16*128的矩阵，A B矩阵都是行主序
gemini 3pro不推荐用m8n8k4，是因为m8n8k4是volta架构的第一代tensorcore，而m16n8k16是apmere架构的第三代tensorcore，m16n8k16相对而言会更好（具体我没验证过）
正常来讲，C=A*B C shape 8*250000, A shape 8*128, B shape 128*250000
这里使用C^T = B^T * A^T的方案 C shape 250000*8, A shape 128*8, B shape 250000*128
然后ldmatrix载入A矩阵的时候使用x2，不带trans的，这样载入到reg的其实是16*8的A^t
对于B矩阵，由于是行主序(250000*128)，相较于128*250000本来就是带了转置的，所以直接用ldmatrix x4载入就行
算出来的C是带了转置的结果，这个需要注意
A矩阵使用一次cp async之后，直接存放在smem或者寄存器中就行
B矩阵重复不断使用cp async，每次cp async会变换N维度做载入
下面将16作为B的N的一个大 tile，一个block会处理H个大 tile，H的大小关系到mainloop能循环多少次

单个block寄存器和smem使用(一个block有32个thread):
对于A矩阵，smem占用8*128*2byte，reg占用8*128*2byte，reg总共占用8*128*2/4个uint32寄存器
对于B矩阵，设stage为3，则占用16*128*3*2byte，reg占用16*16*2/4个uint32寄存器
关于C矩阵，对于一个B的大 tile，计算的结果为16*8（C^T），如果处理H个大 tile，会占用H*16*8*2/4个uint32寄存器
一般希望H较大，这样mainloop能够占据更多时间，使得prologue和epilogue的时间占用比例变小
假设H为100，则会占用6400个uint32寄存器，显然太多了
所以C的结果只能使用stmatrix存放到smem中
则C矩阵只用占用16*8*2/4个寄存器，而smem占用为H*16*8*2byte，设H=100，则占用25600byte

对于2的方案，还有另一种改进方案可选：一个block有256个thread
之所以是改进方案，是因为，对于一个block 32个thread的方案，由于cp async对于单个线程最多只能读取16 byte，如果想要将16*128half的b矩阵完全读入的话，需要8次cp async之后，直接存放在smem或者寄存器中就行
如果希望发射一次cp async就将16*128完全读入的话，需要256个线程，即总共8个warp，每个warp算一个m16n8k16的mma，这样每个thread在mainloop中，一个8*128和16*128的矩阵乘只需要发射一次mma，这样每个block的并行度灰比较高
对于一个block有256个thread时，一个block的smem和reg的使用量：
reg：
A矩阵：8*128*2/4个uint32，B矩阵：16*128*2/4个uint32，C矩阵：16*8*2/4个uint32寄存器
smem：
A矩阵：8*128*2，B矩阵：16*128*2*stage，C矩阵：8*16*2*H
这里设stage为3，H为100
则一个block总寄存器使用量为：8*128*2/4 + 16*128*2/4 + 16*8*2/4 = 1600个uint32
一个block总的smem使用量为：8*128*2 + 16*128*2*3 + 8*16*2*100 = 39936byte
则一个sm上可以启动的block数量为
min(65536/1600, 167936/39936) = min(40.96, 4.2) = 4

还有一种选择是

