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

### 任务一：基于TENET的relation-centric数据流评估

> 难点：理解relation-centric数据流评估过程、ISL库学习
>
> 难度：中级

TENET github仓库地址：https://github.com/pku-liang/TENET

TENET代码中使用ISL库来进行整数集合操作，ISL库的使用方法可参考[手册](https://libisl.sourceforge.io/manual.pdf)。TENET代码中的data文件夹含有PE阵列、数据流mapping、算子statement的模板。

**你需要做：**

1. 针对下面的GEMM算子，自行设计数据流mapping，使用TENET评估将算子送入16x16的2D systolic PE阵列上执行的PE平均使用率，并仿照TENET论文中的Figure3简要描述数据流动过程，说明使用了哪种复用。

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

1. 将2D PE阵列的数据流扩展到16x16x16的3D systolic PE阵列，设计下面GEMM算子的数据流mapping，经TENET评估后的PE平均使用率达到PE总量的50%以上，比如，当Active PE Num为4096时，Average Active PE Num大于等于2048。仿照TENET论文中的Figure3简要描述数据流动过程。
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
   
1. （选做）由于ISL库的问题，目前版本的TENET代码在计算数据重用时有bug，在一些明显有数据重用的数据流策略下，计算得到的totalvolume和uniquevolume是一样的，也即评估的结果显示没有数据重用。请复现该bug，并尝试自定义数据结构，实现类似于ISL库中的整数集操作，解决该bug。

### 任务二：基于Rubick的数据流设计空间探索

> 难点：数据流建模与线性代数的关系
>
> 难度：中级
>

TENET以relation-centric的方式建模数据流并进行评估，不过缺乏对数据流设计空间进行探索的能力，而探索过程人工来做的话耗时耗力，因此需要自动化探索框架。Rubick是基于TENET实现的数据流设计空间探索框架，请仔细阅读[Rubick论文](https://ieeexplore.ieee.org/document/10247743)和代码（代码位于夏令营项目仓库的rubick-main文件夹内），学习数据流设计空间探索的IR建模抽象。


1. 回答问题，请结合线性代数的知识简要解释：
   - Rubick论文中使用方向向量来表征access entry的数学合理性是什么？
   - 为什么2D PE阵列的access entry类型初始有63种，又为什么可以剪枝成14种？
2. 设计一个自动剪枝程序，对2D的access entry类型进行剪枝，可以不考虑对称线性空间和硬件实现合理性（2D access entry此时剪枝后的结果应为17种）。由于2D和3D的access entry类型剪枝策略一致，剪枝程序需能对3D的access entry类型进行剪枝。请阐述程序设计思路。
3. （选做）简要分析Rubick代码中数据流设计空间探索的步骤，以及当中涉及的优化算法。
4. （选做）可以使用任意HDL或HLS，设计一个systolic加速器，请阐述设计思路。

