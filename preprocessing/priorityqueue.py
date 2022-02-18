# -*- coding:utf-8 -*-
# @Time : 2022/2/18 10:51 下午
# @Author : huichuan LI
# @File : priorityqueue.py
# @Software: PyCharm
import heapq
from copy import copy
import numpy as np

class PQNode(object):
    def __init__(self, key, val, priority, entry_id, **kwargs):
        """A generic node object for holding entries in :class:`PriorityQueue`"""
        self.key = key
        self.val = val
        self.entry_id = entry_id
        self.priority = priority

    def __repr__(self):
        fstr = "PQNode(key={}, val={}, priority={}, entry_id={})"
        return fstr.format(self.key, self.val, self.priority, self.entry_id)

    def to_dict(self):
        """Return a dictionary representation of the node's contents"""
        d = self.__dict__
        d["id"] = "PQNode"
        return d

    def __gt__(self, other):
        if not isinstance(other, PQNode):
            return -1
        if self.priority == other.priority:
            return self.entry_id > other.entry_id
        return self.priority > other.priority

    def __ge__(self, other):
        if not isinstance(other, PQNode):
            return -1
        return self.priority >= other.priority

    def __lt__(self, other):
        if not isinstance(other, PQNode):
            return -1
        if self.priority == other.priority:
            return self.entry_id < other.entry_id
        return self.priority < other.priority

    def __le__(self, other):
        if not isinstance(other, PQNode):
            return -1
        return self.priority <= other.priority


class PriorityQueue:
    def __init__(self, capacity, heap_order="max"):
        """
        A priority queue implementation using a binary heap.
        Notes
        -----
        A priority queue is a data structure useful for storing the top
        `capacity` largest or smallest elements in a collection of values. As a
        result of using a binary heap, ``PriorityQueue`` offers `O(log N)`
        :meth:`push` and :meth:`pop` operations.
        Parameters
        ----------
        capacity: int
            The maximum number of items that can be held in the queue.
        heap_order: {"max", "min"}
            Whether the priority queue should retain the items with the
            `capacity` smallest (`heap_order` = 'min') or `capacity` largest
            (`heap_order` = 'max') priorities.
        """
        assert heap_order in ["max", "min"], "heap_order must be either 'max' or 'min'"
        self.capacity = capacity
        self.heap_order = heap_order

        self._pq = []
        self._count = 0
        self._entry_counter = 0

    def __repr__(self):
        fstr = "PriorityQueue(capacity={}, heap_order={}) with {} items"
        return fstr.format(self.capacity, self.heap_order, self._count)

    def __len__(self):
        return self._count

    def __iter__(self):
        return iter(self._pq)

    def push(self, key, priority, val=None):
        """
        Add a new (key, value) pair with priority `priority` to the queue.
        Notes
        -----
        If the queue is at capacity and `priority` exceeds the priority of the
        item with the largest/smallest priority currently in the queue, replace
        the current queue item with (`key`, `val`).
        Parameters
        ----------
        key : hashable object
            The key to insert into the queue.
        priority : comparable
            The priority for the `key`, `val` pair.
        val : object
            The value associated with `key`. Default is None.
        """
        if self.heap_order == "max":
            priority = -1 * priority

        item = PQNode(key=key, val=val, priority=priority, entry_id=self._entry_counter)
        heapq.heappush(self._pq, item)

        self._count += 1
        self._entry_counter += 1

        while self._count > self.capacity:
            self.pop()

    def pop(self):
        """
        Remove the item with the largest/smallest (depending on
        ``self.heap_order``) priority from the queue and return it.
        Notes
        -----
        In contrast to :meth:`peek`, this operation is `O(log N)`.
        Returns
        -------
        item : :class:`PQNode` instance or None
            Item with the largest/smallest priority, depending on
            ``self.heap_order``.
        """
        item = heapq.heappop(self._pq).to_dict()
        if self.heap_order == "max":
            item["priority"] = -1 * item["priority"]
        self._count -= 1
        return item

    def peek(self):
        """
        Return the item with the largest/smallest (depending on
        ``self.heap_order``) priority *without* removing it from the queue.
        Notes
        -----
        In contrast to :meth:`pop`, this operation is O(1).
        Returns
        -------
        item : :class:`PQNode` instance or None
            Item with the largest/smallest priority, depending on
            ``self.heap_order``.
        """
        item = None
        if self._count > 0:
            item = copy(self._pq[0].to_dict())
            if self.heap_order == "max":
                item["priority"] = -1 * item["priority"]
        return item
