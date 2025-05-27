[TOC]

<div STYLE="page-break-after: always;"></div>

## 滑动窗口, 双指针

#### 滑动窗口 : 

##### [3. 无重复字符的最长子串 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        if not s: return 0
        left = 0
        MAX = 1
        pos_val = {}
        for right, c in enumerate(s):
            if s[right] in pos_val and pos_val[s[right]] >= left:
                left = pos_val[s[right]] + 1
            pos_val[s[right]] = right  # pos_val : left ~ right
            if right - left + 1 > MAX:
                MAX = right - left + 1
        return MAX
```

#### 双指针 : 

##### [11. 盛最多水的容器 - 力扣（LeetCode）](https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-100-liked)

``` python
class Solution(object):
    def maxArea(self, height):
        left, right = 0, len(height) - 1
        MAX = 0
        while left <= right:
            if height[left] <= height[right]:
                MAX = max(MAX, (right - left) * height[left])
                left += 1
            else:
                MAX = max(MAX, (right - left) * height[right])
                right -= 1
        return MAX
```

#### 快慢指针 (Floyd's Tortoise and Hare Algorithm)

求链表中点, 判断链表是否有圈

#### 单调栈 :

##### [739. 每日温度 - 力扣（LeetCode）](https://leetcode.cn/problems/daily-temperatures/?envType=problem-list-v2&envId=monotonic-stack)

``` python
class Solution(object):
    def dailyTemperatures(self, temperatures):
        st = []
        res = [0] * len(temperatures)
        for i, t in enumerate(temperatures):
            if not st:
                st.append(i)
            else:
                while st and temperatures[st[-1]] < t:
                    res[st[-1]] = i - st[-1]
                    st.pop()
                st.append(i)
        return res
```

##### [84. 柱状图中最大的矩形 - 力扣（LeetCode）](https://leetcode.cn/problems/largest-rectangle-in-histogram/description/?envType=study-plan-v2&envId=top-100-liked)

Key : 如何遍历?

Solution : 例如 `[3, 1, 4, 1, 5, 9, 2, 6]`

`[3]` => `[1]` => `[1,4]` => `[1]` => `[1,5]` =>  `[1,5,9]` <font : color = red>=></font> `[1,2]` => `[1,2,6]`

其中红色的一步, 每次枚举最右侧是 9 的矩形 (`[9]`, `[5,9]`)  (不会枚举 `[1,5,9]`, 因为 `[4,9,2,...]` 之后会枚举到的)

其中在 `height` 末尾加 `0` 是为了保证最后把 `[1,2,6]` 完整的枚举一遍 (`[6]`, `[2,6]`, `[1,5,9,2,6]`)

```python
class Solution(object):
    def largestRectangleArea(self, heights):
        heights.append(0)
        st = []
        MAX = 0
        for i in range(len(heights)):
            while st and heights[st[-1]] > heights[i]:
                h = heights[st.pop()]
                w = i if not st else i - st[-1] - 1
                MAX = max(MAX, h * w)
            st.append(i)
        return MAX
```

##### [85. 最大矩形 - 力扣（LeetCode）](https://leetcode.cn/problems/maximal-rectangle/)

```python
class Solution(object):
    def maximalColum(self, col):
        col.append(0)
        st = []
        MAX = 0
        for i, x in enumerate(col):
            while st and col[st[-1]] > x:
                if len(st) >= 2:
                    MAX = max(MAX, col[st[-1]] * (i - st[-2] - 1))
                else:
                    MAX = max(MAX, col[st[-1]] * i)
                st.pop()
            st.append(i)
        return MAX
    def maximalRectangle(self, matrix):
        m, n = len(matrix), len(matrix[0])
        pre = [0] * n
        MAX = 0
        for i in range(m):
            for j in range(n):
                pre[j] = pre[j] + 1 if matrix[i][j] == "1" else 0
            MAX = max(MAX, self.maximalColum(pre.copy()))
        return MAX
```

#### 单调队列 : 

##### [239. 滑动窗口最大值 - 力扣（LeetCode）](https://leetcode.cn/problems/sliding-window-maximum/description/?envType=study-plan-v2&envId=top-100-liked)

```python
from collections import deque

class Solution(object):
    def maxSlidingWindow(self, nums, k):
        dq = deque([])
        res = []
        for i, x in enumerate(nums):
            while dq and dq[0] <= i - k:
                dq.popleft()
            while dq and nums[dq[-1]] <= x:
                dq.pop()
            dq.append(i)
            if i >= k - 1:
                res.append(nums[dq[0]])
        return res
```

## Stack

##### [中序表达式转后序表达式](http://cs101.openjudge.cn/practice/24591/)

以下是 Shunting Yard 算法的基本步骤：

1. 初始化运算符栈和输出栈为空. 
2. 从左到右遍历中缀表达式的每个符号. 
   - 如果是操作数(数字), 则将其添加到输出栈. 
   - 如果是左括号, 则将其推入运算符栈. 
   - 如果是运算符：
     - 如果运算符的优先级大于运算符栈顶的运算符, 或者运算符栈顶是左括号, 则将当前运算符推入运算符栈. 
     - 否则, 将运算符栈顶的运算符弹出并添加到输出栈中, 直到满足上述条件(或者运算符栈为空). 
     - 将当前运算符推入运算符栈. 
   - 如果是右括号, 则将运算符栈顶的运算符弹出并添加到输出栈中, 直到遇到左括号. 将左括号弹出但不添加到输出栈中. 
3. 如果还有剩余的运算符在运算符栈中, 将它们依次弹出并添加到输出栈中. 
4. 输出栈中的元素就是转换后的后缀表达式. 

```python
opr_pri = {"+" : 1, "-" : 1, "*" : 2, "/" : 2, "(" : 3, ")" : 3}

def find_num(s : str, i : int) -> int:
    # e.g. find_num("1.0+2.5", 0) = 3
    while i < len(s) and s[i] not in opr_pri:
        i += 1
    return i

def trans() -> list:
    s, i = input(), 0
    res, opr_st = [], []
    while i < len(s):
        if s[i] in opr_pri:
            if s[i] == "(":
                opr_st.append(s[i])
            elif s[i] == ")":
                while opr_st and opr_st[-1] != "(":
                    res.append(opr_st.pop())
                opr_st.pop()
            else:
                while opr_st and opr_st[-1] != "(" and opr_pri[s[i]] <= opr_pri[opr_st[-1]]:
                    res.append(opr_st.pop())
                opr_st.append(s[i])
            i += 1
        else:
            j = find_num(s, i)
            res.append(s[i : j])
            i = j
    while opr_st:
        res.append(opr_st.pop())
    return res

n = int(input())

for _ in range(n):
    print(*trans(), sep = " ")
```

## 排序

##### Merge Sort, [OpenJudge - 07622:求排列的逆序数](http://cs101.openjudge.cn/practice/07622/)

```python
def merge_count(arr1, arr2):
    cnt, j = 0, 0
    for x in arr1:
        while j < len(arr2) and arr2[j] < x:
            j += 1
        cnt += j
    res, i, j = [], 0, 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            res.append(arr1[i]); i += 1
        else:
            res.append(arr2[j]); j += 1
    return res + arr1[i:] + arr2[j:], cnt

def sortArray(nums):
    if not nums or len(nums) == 1:
        return nums, 0
    mid = len(nums) // 2
    arr1, sum1 = sortArray(nums[:mid])
    arr2, sum2 = sortArray(nums[mid:])
    arr, cnt = merge_count(arr1, arr2)
    return arr, sum1 + sum2 + cnt
```




## Linked List

##### 引用与赋值

```python
# 定义链表节点类
class ListNode:
    def __init__(self, val, next = None):
        self.val = val
        self.next = next
    def __str__(self):
        return f"ListNode({self.val} -> {self.next.val})"

d = ListNode(4)
c = ListNode(3, d)
b = ListNode(2, c)
a = ListNode(1, b)
```

1. ```python
   # Example 1 : `prev` 和 `curr` 指向相同的节点, 修改 `prev` 后 `curr` 不受影响
   prev = a
   curr = prev
   prev = b
   print(curr == a, a) # output : True ListNode(1 -> 2)
   ```

2. ```python
   # Example 2 : `curr` 指向 `a.next` (i.e. `b`), 修改 `prev` 后 `curr` 不受影响
   prev = a
   curr = prev.next
   prev = c
   print(curr == b, b) # output : True ListNode(2 -> 3)
   ```

3. ```python
   # Example 3 : `curr` 指向 `a`, 修改 `a.val`, `curr.val` 也受影响
   curr = a
   a.val = 0
   print(curr) # output : ListNode(0 -> 2)
   ```

4. ```python
   # Example 4 : `prev` 和 `curr` 指向相同对象 `a`, 修改 `prev.val`, `curr.val` 也受影响
   prev = a
   curr = a
   prev.val = 0
   print(curr) # output : ListNode(0 -> 2)
   ```

5. ```python
   # Example 5 : `curr` 指向 `a`, 修改 `a.next`, `curr.next` 也受影响
   prev = a
   curr = prev
   prev.next = c
   print(curr) # output : ListNode(0 -> 3)
   ```

引用变更不会同步, 赋值变更 ( `prev.next = ...` 或者 `prev.val = ...`) 会同步



[206. 反转链表 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-linked-list/description/)

```python
class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class Solution(object):
    def reverseList(self, head):
        pre = None
        curr = head
        while curr:
            curr_next = curr.next
            curr.next = pre
            pre = curr
            curr = curr_next
        return pre
```

## Tree

```python
class Tree():
    def __init__(self, val = 0, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right
```

##### 手搓Heapq

此略

**并查集 Disjoint Set**

[OpenJudge - 02524:宗教信仰](http://cs101.openjudge.cn/practice/02524/)

```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    x = find(x)
    y = find(y)
    parent[x] = y

case = 0
while True:
    case += 1
    n, m = map(int, input().split())
    if n == 0:
        break
    parent = [i for i in range(n + 1)]
    for _ in range(m):
        x, y = map(int, input().split())
        union(x, y)
    for i in range(1, n + 1):
        find(i)
    parent_set = set(parent)
    print(f"Case {case}: {len(parent_set) - 1}")
```

## Graph

##### 拓扑排序 (可用于判断有向图中有无环)

Kahn, 时间复杂度 $O(V + E)$

```python
def topological_sort(graph : Dict[str : List[str]]):
    in_degree = defaultdict(int)
    res, que = [], deque()
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    for u in graph:
        if in_degree[u] == 0:
            que.append(u)
    while que:
        u = que.popleft()
        res.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                que.append(v)
    if len(res) == len(graph):
        return res
    else:
        return None # have a cycle
```

##### 最短路径

- **Dijkstra**

  key : 每个点一进一出, 但要求图无负权边

- **Bellman-Ford** $O(VE)$

  ```python
  def bellman_ford(graph, V, source):
      dist = [float('inf')] * V # 初始化距离
      dist[source] = 0
      for _ in range(V - 1): # 松弛 V-1 次
          for u, v, w in graph:
              if dist[u] != float('inf') and dist[u] + w < dist[v]:
                  dist[v] = dist[u] + w
      for u, v, w in graph: # 检测负权环
          if dist[u] != float('inf') and dist[u] + w < dist[v]:
              print("图中存在负权环")
              return None
      return dist
  
  edges = [(0, 1, 5), (0, 2, 4), (1, 3, 3), (2, 1, 6), (3, 2, -2)] # 图是边列表，每条边是 (起点, 终点, 权重)
  V, source = 4, 0 # V 总点数, source 起点
  print(bellman_ford(edges, V, source))
  ```

- **SPFA**

  ```python
  from collections import deque
  
  def spfa(adj, V, source):
      dist = [float('inf')] * V # 初始化距离
      dist[source] = 0
      in_queue = [False] * V # 初始化入队状态
      in_queue[source] = True
      count = [0] * V # 初始化松弛次数
      queue = deque([source])
      while queue:
          u = queue.popleft()
          in_queue[u] = False # in_queue 相当于存储 set(queue)
          for v, w in adj[u]:
              if dist[u] + w < dist[v]:
                  dist[v] = dist[u] + w
                  if in_queue[v] == False:
                      queue.append(v)
                      in_queue[v] = True
                      count[v] += 1
                      if count[v] > V:
                          print("图中存在负权环")
                          return None
      return dist
  
  adj = [[(1, 5), (2, 4)], [(3, 3)], [(1, 6)], [(2, -2)]] # 图的邻接表表示
  V, source = 4, 0 # V 总点数, source 起点
  print(spfa(agj, V, source))
  ```

- **Floyd-Warshall **$O(V^3)$

  ```python
  def floyd_warshall(graph : Dict):
      n = len(graph)
      dist = [[float('inf')] * n for _ in range(n)]
      for i in range(n):
          for j in range(n):
              if i == j:
                  dist[i][j] = 0
              elif j in graph[i]:
                  dist[i][j] = graph[i][j]
      for k in range(n):
          for i in range(n):
              for j in range(n):
   	           dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
      return dis
  ```

##### 最小生成树

- **Prim**, $O(V^2)$, 适用于稠密图

  不断往MST中添加Vertex (greedy思想, 选距离 *现有MST* 权值最小的Vertex)

  ```python
  def prim(n, matrix : List[List[int]]):
      MST, low = set(), [float("inf")] * n # low[k] 表示当前 MST 距离 k 点的最小权值.
      low[0], tot = 0, 0
      for _ in range(n):
          new, MIN = 0, float("inf")
          for i, dis in enumerate(low):
               if i not in MST and dis < MIN:
                  new, MIN = i, dis
          MST.add(new)
          tot += MIN
          for i in range(n):
              if i not in MST:
                  low[i] = min(low[i], matrix[i][new]) # 更新新版 MST 距离 k 点的最小权值.
      return tot
  ```

- ```python
  class DisjointSet:
      def __init__(self, num_vertices):
          self.parent = list(range(num_vertices))
          self.rank = [0] * num_vertices
      def find(self, x):
          if self.parent[x] != x:
              self.parent[x] = self.find(self.parent[x])
          return self.parent[x]
      def union(self, x, y):
          root_x = self.find(x)
          root_y = self.find(y)
          if root_x != root_y:
              if self.rank[root_x] < self.rank[root_y]:
   	           self.parent[root_x] = root_y
              elif self.rank[root_x] > self.rank[root_y]:
                  self.parent[root_y] = root_x
              else:
                  self.parent[root_x] = root_y
                  self.rank[root_y] += 1
  
  def kruskal(graph):
      num_vertices = len(graph)
      edges = [] # 构建边集
      for i in range(num_vertices):
          for j in range(i + 1, num_vertices):
              if graph[i][j] != 0:
                  edges.append((i, j, graph[i][j]))    
      edges.sort(key=lambda x: x[2]) # 按照权重排序
      disjoint_set = DisjointSet(num_vertices) # 初始化并查集
      minimum_spanning_tree = [] # 构建最小生成树的边集
      for edge in edges:
          u, v, weight = edge
          if disjoint_set.find(u) != disjoint_set.find(v):
              disjoint_set.union(u, v)
              minimum_spanning_tree.append((u, v, weight))
      return minimum_spanning_tree
  ```

---------

##### 强连通 <mark>sorry</mark>

