# Assignment #5: 链表、栈、队列和归并排序

Updated 1348 GMT+8 Mar 17, 2025

2025 spring, Complied by **袁奕 2400010766 数院**



> **说明：**
>
> 1. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 2. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 3. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### LC21.合并两个有序链表

linked list, https://leetcode.cn/problems/merge-two-sorted-lists/

思路：



代码：

```python
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        if not list1:
            return list2
        if not list2:
            return list1
        if list1.val <= list2.val:
            head1 = ListNode(list1.val)
            head2 = self.mergeTwoLists(list1.next, list2)
        else:
            head1 = ListNode(list2.val)
            head2 = self.mergeTwoLists(list2.next, list1)
        head1.next = head2
        return head1
```

[21. 合并两个有序链表 - 力扣（LeetCode）](https://leetcode.cn/problems/merge-two-sorted-lists/submissions/608219820/)

![](https://cdn.jsdelivr.net/gh/yuanyi-350/image/20250318195849.png)





### LC234.回文链表

linked list, https://leetcode.cn/problems/palindrome-linked-list/

<mark>请用快慢指针实现。</mark>

询问 ChatGPT, 快慢指针 : Floyd's Tortoise and Hare Algorithm

**Hare（兔子，快指针）** 每次走 **两步**

**Tortoise（乌龟，慢指针）** 每次走 **一步**

🐇走完全程, 🐢走完一半, 可以用来寻找中点. 

以及上次作业的判断是否有环也是这样.

```python
class Solution(object):
    def mid(self, head):
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        if fast:
            return slow, slow.next
        else:
            return slow, slow
    def reverse(self, head):
        pre = None
        curr = head
        while curr:
            curr_next = curr.next
            curr.next = pre
            pre = curr
            curr = curr_next
        return pre
    def isPalindrome(self, head):
        mid0, mid1 = self.mid(head)
        head1 = self.reverse(mid1)

        while head1:
            if head1.val != head.val:
                return False
            head1 = head1.next
            head = head.next

        if not head.next or not head.next.next:
            return True
        return False
```

![](https://cdn.jsdelivr.net/gh/yuanyi-350/image/20250319084118.png)





### LC1472.设计浏览器历史记录

doubly-lined list, https://leetcode.cn/problems/design-browser-history/

<mark>请用双链表实现。</mark>



代码：

```python
class ListNode:
    def __init__(self, val, pre = None, next = None):
        self.val = val
        self.pre = pre
        self.next = next

class BrowserHistory(object):
    def __init__(self, homepage):
        self.head = ListNode(homepage)
        self.curr = self.head
    def visit(self, url):
        new = ListNode(url, self.curr)
        self.curr.next = new
        self.curr = new
    def back(self, steps):
        for _ in range(steps):
            if self.curr.pre:
                self.curr = self.curr.pre
            else:
                return self.curr.val
        return self.curr.val
    def forward(self, steps):
        for _ in range(steps):
            if self.curr.next:
                self.curr = self.curr.next
            else:
                return self.curr.val
        return self.curr.val
```



![](https://cdn.jsdelivr.net/gh/yuanyi-350/image/20250319113048.png)





### 24591: 中序表达式转后序表达式

stack, http://cs101.openjudge.cn/practice/24591/

思路：



代码：

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



![](https://cdn.jsdelivr.net/gh/yuanyi-350/image/20250319110846.png)





### 03253: 约瑟夫问题No.2

queue, http://cs101.openjudge.cn/practice/03253/

<mark>请用队列实现。</mark>



代码：

```python
from collections import deque

while True:
    n, p, m = map(int, input().split())
    if n == 0:
        break
    que = deque(range(1, n + 1))
    res = []
    # 先考虑从 1 开始报数, 结果统一旋转 p - 1
    while que:
        for _ in range(m - 1):
            que.append(que.popleft())
        res.append(que.popleft())
    print(*[(i + p - 2) % n + 1 for i in res], sep = ",")
```

![](https://cdn.jsdelivr.net/gh/yuanyi-350/image/20250318201649.png)





### 20018: 蚂蚁王国的越野跑

merge sort, http://cs101.openjudge.cn/practice/20018/

思路：本质求逆序对



代码：

```python
def merge_count(arr1, arr2):
    cnt, j = 0, 0
    for x in arr1:
        while j < len(arr2) and arr2[j] <= x:
            j += 1
        cnt += len(arr2) - j
    res, i, j = [], 0, 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            res.append(arr1[i]); i += 1
        else:
            res.append(arr2[j]); j += 1
    return res + arr1[i:] + arr2[j:] ,cnt

def sortArray(nums):
    if not nums or len(nums) == 1:
        return nums, 0
    mid = len(nums) // 2
    arr1, sum1 = sortArray(nums[:mid])
    arr2, sum2 = sortArray(nums[mid:])
    arr, cnt = merge_count(arr1, arr2)
    return arr, sum1 + sum2 + cnt

n = int(input())
nums = [int(input()) for _ in range(n)]
print(sortArray(nums)[1])
```

![](https://cdn.jsdelivr.net/gh/yuanyi-350/image/20250319152901.png)



## 2. 学习总结和收获

##### 总结了链表引用与赋值的cheating sheet

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





##### 做的比较有意义 ~~(困难的)~~ 题目 : 

[25. K 个一组翻转链表 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-nodes-in-k-group/description/)

```python
class Solution(object):
    def len(self, node, k):
        # e.g. k = 2, 1 -> 2 -> 3 -> 4, len(3) = 2
        if not node:
            return False
        for i in range(k - 1):
            if not node.next:
                return False
            node = node.next
        return True
    def reverse_next_k(self, head, k):
        #e.g. 1 -> 2 -> 3 -> 4 -> 5, k = 3, return 3, 1
        #e.g. 1 -> 2, k = 3, return None None
        if not self.len(head, k):
            return None, None
        pre = None
        curr = head
        cnt = 1
        while curr and cnt <= k:
            curr_next = curr.next
            curr.next = pre
            pre = curr
            curr = curr_next
            cnt += 1
        head.next = curr
        return pre, head

    def reverseKGroup(self, head, k):
        start, end = self.reverse_next_k(head, k)
        head = start
        while end and end.next:
            new_start, new_end = self.reverse_next_k(end.next, k)
            if new_start:
                end.next = new_start
                start, end = new_start, new_end
            else:
                break
        return head
```



[295. 数据流的中位数 - 力扣（LeetCode）](https://leetcode.cn/problems/find-median-from-data-stream/description/)

```python
from heapq import heappush, heappop

class Heap:
    def __init__(self, is_max = True):
        self.hp = []
        self.is_max = is_max
    def push(self, ele):
        heappush(self.hp, -ele if self.is_max else ele)
    def pop(self):
        if self.hp:
            ele = heappop(self.hp)
            ele = - ele if self.is_max else ele
            return ele
    def peek(self):
        if self.is_max:
            return - self.hp[0] if self.hp else None
        else:
            return self.hp[0] if self.hp else None

class MedianFinder(object):
    def __init__(self):
        self._min = Heap()
        self._max = Heap(False)
        self.mid = None
    def update(self):
        self.balance()
        if len(self._min.hp) + 1 == len(self._max.hp):
            self.mid = self._max.peek()
        if len(self._min.hp) == len(self._max.hp) + 1:
            self.mid = self._min.peek()
        if len(self._min.hp) == len(self._max.hp):
            if self._min.hp:
                self.mid = (self._min.peek() + self._max.peek()) / 2.0
            else:
                self.mid = None
    def balance(self):
        if len(self._min.hp) + 1 < len(self._max.hp):
            self._min.push(self._max.pop())
        elif len(self._min.hp) > len(self._max.hp) + 1:
            self._max.push(self._min.pop())
    def addNum(self, num):
        self.update()
        if self.mid == None:
            self.mid = num
        if num <= self.mid:
            self._min.push(num)
        else:
            self._max.push(num)
    def findMedian(self):
        self.update()
        return self.mid
```

