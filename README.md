# 《数据分析咖哥十话》
这个Library是数据分析咖哥十话一书的代码库和勘误表，以及一些值得讨论的东西。


本书购买链接：https://item.jd.com/13335199.html

![](https://img-blog.csdnimg.cn/bce1b5ef95f342fdbd9808bc3eb563b5.png)

# 一刷勘误（已经在2022年9月的第二次印刷后改正）
## 整体
有些地方的RMF均应改为RFM。
## P020
 瘦狗是指市场占有率低及市场占有率低的业务。应改为：瘦狗是指市场占有率低及销售增长率低的业务 。

## P040
原输出图中单身的14和有朋友的19应该对调。

正确的图：
![](https://img-blog.csdnimg.cn/7d42afe3ce034ac387e6babc78caaeb8.png)

## P052
userList =["咖哥", "马总", "小冰", "小雪"] #创建元组

改为

userList = ("咖哥", "马总", "小冰", "小雪") #创建元组

中括号改成小括号，才是元组。

## P057
Out: 2.29999999999999998

改为

Out:0.72587994

这是为了和P56页的npArray05内容保持一致。
## P057
![](https://img-blog.csdnimg.cn/1d80e778092041bb8869bb4764142010.png)

npArray02数组中第四列的前三个元素，

改为

npArray02数组中**第二行**的前三个元素。

文章和代码都要改。
## P149
1/5 = 0.5改为0.2

3/4 + 1/4 = 1 改为 4/5 + 1/5 = 1

下面一行的 1/2 和 1/4应该对调。

![](https://img-blog.csdnimg.cn/7db8cb5efb1e4037bd6b29d4cf7cb3ba.png)

## P246

也许得了感冒，改为，也许长了痘痘。


# 讨论

## 第三章 预测用户的LTV

有小伙伴在异步图书的勘误区发表了下面的建议：

![](https://img-blog.csdnimg.cn/7e28cce8124d44249a19c51fb3ead75d.jpeg)

这是一个很认真的建议。根据不同的业务需求，的确是可以这样构建数据集的。严格来说，应该从每个用户的第一次购买日期算起，这个建议很好，因为我们希望根据头三个月的信息预测后续的LTV。不过，我们收集前三个月的数据时，并不是很确定每个用户都会持续购物一年。因此，我个人认为不需要选择持续购物一年的用户，即使客户流失，那么模型也应该吸收这些信息，从而具备判断这类用户因为可能流失而产生低LTV的可能性。
