#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""


from abc import ABC, abstractmethod

class AbstractAlgorithm(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def run(self, data):
        pass
