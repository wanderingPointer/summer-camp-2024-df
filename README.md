# 数据流项目

## 前置知识学习

数据流（dataflow）指的是将DNN算子送入DNN加速器执行时的计算负载分散和数据移动策略，评价一个数据流策略的标准往往包含数据重用率、计算单元（Processing Element，PE）平均使用率等。为完成后面的任务，建议先学习以下内容：

1. 理解DNN算子、systolic等DNN加速器结构、数据的几种重用方式及对应的抽象硬件实现。
2. 理解数据流的compute-centric、data-centric、relation-centric表征方式及优缺点。
3. 理解数据流评估的数据重用率、PE平均使用率等指标的计算方法。

**参考资料：**

- compute-centric数据流的典型论文：[Timeloop](https://ieeexplore.ieee.org/document/8695666)
- data-centric数据流的典型论文：[MAESTRO](https://dl.acm.org/doi/10.1145/3352460.3358252)
- relation-centric数据流和分析数据流评估指标的典型论文：[TENET](https://dl.acm.org/doi/abs/10.1109/ISCA52012.2021.00062)

## 任务布置

根据能力完成任务，若时间足够可进一步探索拓展内容，编程语言不限。最后需要提交任务报告（Markdown、LaTeX、Word文档皆可），并准备最后阶段的汇报。**可围绕任务设计实验，把一切所思所想和实验结果写进报告；即使时间关系没有完成任务，也建议把所有所做的工作写进报告**。

### 任务零：CPU上执行不同tiling策略的矩阵乘时间对比

> 难点：感受不同tiling策略的矩阵乘时间差别
>
> 难度：初级


矩阵tiling（分块）是一种优化矩阵运算的技术，它通过将大的矩阵乘法操作分解成小块（tiling块），分别对每个小块进行计算，根据硬件支持的并行度来进行并行计算，减少内存访问次数和提高并行性能。本任务旨在让大家感受不同tiling策略下的矩阵乘时间性能差别，提供了一个matrix_tiling程序。matrix_tiling程序中使用了Python的并行计算库进行多线程并行执行，每个线程一次只对一个tiling块进行处理。请大家运行仓库中的matrix_tiling程序（除非设备不支持某线程数量的并行，否则无需更改代码中的线程数列表），绘制矩阵乘执行时间与线程数的关系折线图。

### 任务一：基于TENET的relation-centric数据流评估

> 难点：理解relation-centric数据流评估过程、ISL库学习
>
> 难度：中级

任务零中大家感受了不同tiling策略的矩阵乘在CPU上执行的时间差异，而对于加速器上的矩阵乘法而言，依赖于PE阵列自然的硬件并行结构，不涉及CPU上线程切换、通信等机制带来的开销，具有更优异的性能。下面的任务基于TENET进行加速器上的矩阵乘法数据流评估。

TENET github仓库地址：https://github.com/pku-liang/TENET

TENET代码中使用ISL库来进行整数集合操作，ISL库的使用方法可参考[手册](https://libisl.sourceforge.io/manual.pdf)。TENET代码中的data文件夹含有PE阵列、数据流mapping、算子statement的模板。

**你需要做：**

1. 对于下面的GEMM算子，针对16x16的2D systolic PE阵列设计数据流mapping，mapping满足如下条件：

   - 空间戳的第一、二维分别仅与迭代变量i、k相关。
   - 所有PE都会被使用到。

   之后使用TENET评估算子送入PE阵列上执行的PE平均使用率，并仿照TENET论文中的Figure3简要描述数据流动过程，说明使用了哪种复用。
   $$
   PE平均使用量=\frac{\Sigma每个时钟周期PE的使用数量 }{时钟周期数}\\    PE平均使用率=\frac{PE平均使用量}{PE总量}\times100\%
   $$

   ``` c
   // GEMM算子
   for(int i = 0; i < 128; i++)
   {
       for(int j = 0; j < 64; j++)
       {
           for(int k = 0; k < 32; k++)
               C[i][j] += A[i][k] * B[k][j];
       }
   }
   ```

1. 将2D PE阵列的数据流扩展到16x16x16的3D systolic PE阵列，设计下面GEMM算子的数据流mapping，经TENET评估后的PE平均使用率达到50%以上。仿照TENET论文中的Figure3简要描述数据流动过程。

   ``` c
   // GEMM算子
   for(int i = 0; i < 512; i++)
   {
       for(int j = 0; j < 512; j++)
       {
           for(int k = 0; k < 512; k++)
               C[i][j] += A[i][k] * B[k][j];
       }
   }
   ```

### 任务二：基于Rubick的数据流设计空间探索

> 难点：数据流建模与线性代数的关系
>
> 难度：中级

TENET以relation-centric的方式建模数据流并进行评估，不过缺乏对数据流设计空间进行探索的能力，而探索过程人工来做的话耗时耗力，因此需要自动化探索框架。Rubick是基于TENET实现的数据流设计空间探索框架，请仔细阅读[Rubick论文](https://ieeexplore.ieee.org/document/10247743)和代码（代码位于夏令营项目仓库的rubick-main文件夹内），学习数据流设计空间探索的IR建模抽象。


1. 回答问题，请结合线性代数的知识简要解释：
   - Rubick论文中使用方向向量来表征access entry的数学合理性是什么？
   - 为什么2D PE阵列的access entry类型初始有63种，又为什么可以剪枝成14种？
2. 设计一个自动剪枝程序，对2D的access entry类型进行剪枝，可以不考虑对称线性空间和硬件实现合理性（2D access entry此时剪枝后的结果应为17种）。由于2D和3D的access entry类型剪枝策略一致，剪枝程序需能对3D的access entry类型进行剪枝。请阐述程序设计思路。
3. （选做）简要分析Rubick代码中数据流设计空间探索的步骤，以及当中涉及的优化算法。

### 环境配置注意

1. 运行TENET若出现缺少库文件的问题，可按照https://blog.51cto.com/u_15127691/4347156链接的方法将external/lib中的库添加到库搜索路径中。
2. Rubick的运行环境与TENET基本一致，运行时若出现找不到module问题，请将src目录导入到Python的module搜索路径中。
