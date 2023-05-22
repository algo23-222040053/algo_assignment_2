# algo_assignment_2
- <Building An Asset Trading Strategy> https://www.kaggle.com/code/shtrausslearning/building-an-asset-trading-strategy

## 数据选取
  文章中的研究对象为比特币（12年-20年）的tick数据，这里改为上证指数（00年-20年）的日频数据。
## 目标变量
  我们将短期（窗口）移动平均线 SMA1 和长期（窗口）移动平均线 SMA2 用于创建目标变量信号。
  交易策略如下；其中短期 (SMA1) > 长期 (SMA2)，信号值 = 1（买入），否则设置为 0（卖出）。
  短期 (SMA1) 和长期 (SMA2) 移动平均值分别设置为窗口值 10 和 60，这两个值都是任意的，并且会影响结果，理想情况下需要进行优化研究以找到最佳值值。
![image](https://github.com/algo23-222040053/algo_assignment_2/assets/98448461/2501a491-5554-4315-b00b-afabb79f13f9)

  
